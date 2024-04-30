import os
import io
import numpy as np
import cv2
from PIL import Image
from flask import request, send_file, json, jsonify, Response
from paddleocr import PaddleOCR, draw_ocr
import pypdfium2 as pdfium
import warnings
warnings.filterwarnings("ignore")

class OcrMethodDrawbox:
    ocr = None

    def __init__(self):
        if OcrMethodDrawbox.ocr is None:
            OcrMethodDrawbox.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def extract_coordinates(self, result):
        detected_text_and_boxes = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                text = line[1][0]
                points = line[0]
                x1, y1 = points[0]
                x2, y2 = points[1]
                x3, y3 = points[2]
                x4, y4 = points[3]
                detected_text_and_boxes.append({
                    'text': text,
                    'x1,y1': [x1, y1],
                    'x2,y2': [x2, y2],
                    'x3,y3': [x3, y3],
                    'x4,y4': [x4, y4]
                })
        return detected_text_and_boxes

    def process_image(self, image_data):
        try:
            result = OcrMethodDrawbox.ocr.ocr(image_data)
            if not result:
                raise ValueError("Image OCR result is empty.")
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            boxes = [line[0] for line in result[0]]
            font_path = "latin.ttf"
            im_show = draw_ocr(image, boxes, font_path=font_path)
            pil_im_show = Image.fromarray(im_show)
            image_buffer = io.BytesIO()
            pil_im_show.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            return image_buffer, result
        except Exception as e:
            print(f"Error in process_image: {e}")
            return None, None

    def process_pdf(self, pdf_data):
        file_path = io.BytesIO(pdf_data)
        pdf = pdfium.PdfDocument(file_path)
        n_pages = len(pdf)
        image_path_list = []

        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            pil_image = page.render(scale=3).to_pil()
            image_path = f"image_{page_number + 1}.png"
            pil_image.save(image_path)
            image_path_list.append(image_path)

        processed_images = []
        extracted_coordinates = []

        for image_path in image_path_list:
            image = cv2.imread(image_path)
            result = OcrMethodDrawbox.ocr.ocr(image)
            image_buffer, _ = self.process_image(cv2.imencode(".png", image)[1].tobytes())
            os.remove(image_path)
            processed_images.append(image_buffer)
            extracted_coordinates.extend(self.extract_coordinates(result))

        pdf.close()
        return processed_images, extracted_coordinates

    def boundingbox(self, file):
        file_end = file.filename.endswith(".pdf")

        if file_end == True:
            pdf_data = file.read()
            processed_images, extracted_coordinates = self.process_pdf(pdf_data)
        else:
            image_data = file.read()
            image_buffer, result = self.process_image(image_data)
            extracted_coordinates = self.extract_coordinates(result)
            processed_images = [image_buffer]

        res_image = send_file(processed_images[0], mimetype='image/png')
        return res_image

