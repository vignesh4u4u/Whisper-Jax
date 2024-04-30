import json
from flask import jsonify
import cv2
import numpy as np
import io
import os
from paddleocr import PaddleOCR
import warnings
warnings.filterwarnings("ignore")
import pypdfium2 as pdfium
import cohere
from langchain.llms import CTransformers,HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from ocr.image_to_pattern import HugggingfaceModel,HuggingfaceAccess
os.environ[HugggingfaceModel()]=HuggingfaceAccess()

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
        text_prompt = "<s>[INST] What is your favourite condiment? [/INST]"
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        "[INST] Do you have mayonnaise recipes? [/INST]"
        template = """
        You are an intelligent chatbot. Help the following question with brilliant answers.
        Question: {question}
        Answer:"""
        model1 = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                model_kwargs={"temperature": 0.3,
                                              "seed": 42,
                                              "max_new_tokens": 1500,
                                              "max_length": 1500})

        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=model1)
        response = llm_chain.run(result)
        return response

    def text(self,result):
        try:
            text = " "
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    text += line[1][0] + " "
            input_query = "Please provide the extracted text data in the correct JSON format without causing any JSON decode errors."
            result = text.strip() + "\n\n\n" + input_query
            response = self.final_output(result)
            answer_index = response.find("Answer:")
            if answer_index != -1:
                answer_text = response[answer_index + len("Answer:") + 1:].strip()
                start_index_square = answer_text.find('[')
                start_index_curly = answer_text.find('{')
                if start_index_square != -1 and (start_index_curly == -1 or start_index_square < start_index_curly):
                    start_index = start_index_square
                    end_char = ']'
                elif start_index_curly != -1 and (start_index_square == -1 or start_index_curly < start_index_square):
                    start_index = start_index_curly
                    end_char = '}'
                else:
                    return ("Error: JSON data not found.")
                    extracted_json = None
                end_index = answer_text.rfind(end_char)
                if start_index != -1 and end_index != -1:
                    extracted_json = answer_text[start_index:end_index + 1]
                    return (extracted_json)
                else:
                    return ("Error: JSON data not found.")
            else:
                answer_text = response
                start_index_square = answer_text.find('[')
                start_index_curly = answer_text.find('{')
                if start_index_square != -1 and (start_index_curly == -1 or start_index_square < start_index_curly):
                    start_index = start_index_square
                    end_char = ']'
                elif start_index_curly != -1 and (start_index_square == -1 or start_index_curly < start_index_square):
                    start_index = start_index_curly
                    end_char = '}'
                else:
                    return ("Error: JSON data not found.")
                    extracted_json = None
                end_index = answer_text.rfind(end_char)
                if start_index != -1 and end_index != -1:
                    extracted_json = answer_text[start_index:end_index + 1]
                    return (extracted_json)
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