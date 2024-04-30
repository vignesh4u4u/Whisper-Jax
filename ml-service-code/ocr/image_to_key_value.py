import json
from flask import jsonify
import cv2
import numpy as np
import io
import os
import re
from paddleocr import PaddleOCR
import warnings
warnings.filterwarnings("ignore")
import pypdfium2 as pdfium
import cohere
from langchain.llms import CTransformers,HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from ocr.image_to_pattern import groq_model_name,groq_model_accesess
os.environ[groq_model_name()]=groq_model_accesess()
from groq import Groq

class KeyNameMapping:

    ocr = None
    llm = None
    def __init__(self):
        if KeyNameMapping.ocr is None:
            KeyNameMapping.ocr = PaddleOCR(use_angle_cls=True, lang="en")


    def replace_keys(self,json_data,key_mappings):
        def replace_key_recursive(obj):
            if isinstance(obj, dict):
                keys_to_replace = list(obj.keys())
                for old_key in keys_to_replace:
                    for mapping in key_mappings:
                        if old_key == mapping['oldKey']:
                            new_key = mapping['newKey']
                            obj[new_key] = obj.pop(old_key)
                            break
                for key, value in obj.items():
                    obj[key] = replace_key_recursive(value)
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = replace_key_recursive(obj[i])
            return obj

        replaced_json = replace_key_recursive(json_data)
        return replaced_json

    def final_output(self,result):
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": result}],
            model="mixtral-8x7b-32768", seed=42, max_tokens=2000
        )
        answer_text = chat_completion.choices[0].message.content
        respose = answer_text
        return respose

    def text(self,result):
        try:
            text = " "
            text1 = []
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    text += line[1][0] + "  "
                    text1.append(line[1][0])
                    output = " ".join(text1[0:-1])
            input_query = "Please provide the extracted text data in the proper JSON format, ensuring that the content within the keys and values does not result in any JSON decoding errors."
            result = output.strip() + "\n\n" + input_query
            client = Groq(api_key=os.environ[groq_model_name()])
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": result}],
                model="llama3-8b-8192", seed=42, max_tokens=2000
            )
            answer_text = chat_completion.choices[0].message.content
            respose = answer_text

            if answer_text.find("```"):
                code_blocks = re.findall(r'```(.*?)```', answer_text, re.DOTALL)
                code_text = "\n".join(code_blocks)
                return (code_text)
            else:
                return (answer_text)
        except Exception as e:
            return ({"error":str(e)})

    def process_pdf(self, file_path):
        pdf = pdfium.PdfDocument(file_path)
        n_pages = len(pdf)
        image_path_list = []
        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            pil_image = page.render(scale=3).to_pil()
            image_path = f"image_{page_number + 1}.png"
            pil_image.save(image_path)
            image_path_list.append(image_path)
        detected_text_list = []
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            result = KeyNameMapping.ocr.ocr(image)
            detected_text_list.append(result)
            os.remove(image_path)
        pdf.close()
        return result

    def without_replacekeys(self, image_file):
        try:
            if image_file.filename.endswith(".pdf"):
                file_path = io.BytesIO(image_file.read())
                result = self.process_pdf(file_path)
                output = self.text(result)
                return output
            else:
                image = image_file.read()
                result = KeyNameMapping.ocr.ocr(image)
                if not result:
                    raise ValueError("Image OCR result is empty.")
                output = self.text(result)
                return output
        except Exception as e:
            print(f"Error in process_image: {e}")
            return ({"Json decode error": str(e)})

    def convert(self, image_file, key_mapping):
        try:
            if image_file.filename.endswith(".pdf"):
                file_path = io.BytesIO(image_file.read())
                result = self.process_pdf(file_path)
                output = self.text(result)
                json_data = json.loads(output)
                key_mappings = json.loads(key_mapping)
                final_output = self.replace_keys(json_data, key_mappings)
                return jsonify(final_output)
            else:
                image = image_file.read()
                result = KeyNameMapping.ocr.ocr(image)
                if not result:
                    raise ValueError("Image OCR result is empty.")
                output = self.text(result)
                json_data = json.loads(output)
                key_mappings = json.loads(key_mapping)
                final_output = self.replace_keys(json_data, key_mappings)
                return jsonify(final_output)
        except Exception as e:
            print(f"Error in process_image: {e}")
            return jsonify({"Json decode error": str(e)}), 500