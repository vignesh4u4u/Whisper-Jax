from flask import jsonify
import cv2
import numpy as np
import io
import os
from paddleocr import PaddleOCR
import warnings
warnings.filterwarnings("ignore")
import pypdfium2 as pdfium
import google.generativeai as palm
class ImageToTextConverter:

    ocr = None

    def __init__(self):
        if ImageToTextConverter.ocr is None:
            ImageToTextConverter.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def final_output(self,result):
        defaults = {
            'model': 'models/chat-bison-001',
            'temperature': 0.25,
            'candidate_count': 8,
            'top_k': 40,
            'top_p': 0.95,
        }
        template = """
        You are an intelligent chatbot. Help the following question with brilliant answers.
        Question: {question}
        Answer:"""
        palm.configure(api_key="AIzaSyAUQUIYhpkCrJvSALTiv-NrOLKLUq_9D4U")
        context = ""
        examples = []
        query = result
        response = palm.chat(
            **defaults,
            context=context,
            examples=examples,
            messages=query
        )
        answer = response.last
        return  answer

    def text(self,result):
        detected_text = " "
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                score = line[1][1]
                coordinates_box = line[0]
                detected_text += line[1][0] + "  "
        text = detected_text
        input_query = "please extracted to give the above text data in json format output"
        result = text + "\n\n\n" + input_query
        response = self.final_output(result)
        start_index_square = response.find('[')
        start_index_curly = response.find('{')
        if start_index_square != -1 and (start_index_curly == -1 or start_index_square < start_index_curly):
            start_index = start_index_square
            end_char = ']'
        elif start_index_curly != -1 and (start_index_square == -1 or start_index_curly < start_index_square):
            start_index = start_index_curly
            end_char = '}'
        else:
            print("Error: JSON data not found.")
            extracted_json = None
        end_index = response.rfind(end_char)
        if start_index != -1 and end_index != -1:
            extracted_json = response[start_index:end_index + 1]
            return (extracted_json)
        else:
            print("Error: JSON data not found.")

    def convert(self, image_file):
        if image_file.filename.endswith(".pdf"):
            file_path = io.BytesIO(image_file.read())
            pdf = pdfium.PdfDocument(file_path)
            n_pages = len(pdf)
            image_path_list = []
            for page_number in range(n_pages):
                page = pdf.get_page(page_number)
                pil_image = page.render(scale=3).to_pil()
                image_path = f"image_{page_number + 1}.png"
                pil_image.save(image_path)
                image_path_list.append(image_path)
            for page_number in range(n_pages):
                page = pdf.get_page(page_number)
                pil_image = page.render(scale=3).to_pil()
                image_path = f"image_{page_number + 1}.png"
                pil_image.save(image_path)
                image_path_list.append(image_path)
                detected_text_list = []
                for image_path in image_path_list:
                    image = cv2.imread(image_path)
                    result = ImageToTextConverter.ocr.ocr(image)
                pdf.close()
                os.remove(image_path)
                output = self.text(result)
                return output
        else:
            try:
                image = image_file.read()
                result = ImageToTextConverter.ocr.ocr(image)
                if not result:
                    raise ValueError("Image OCR result is empty.")
                output = self.text(result)
                return output
            except Exception as e:
                print(f"Error in process_image: {e}")
                return None, None
