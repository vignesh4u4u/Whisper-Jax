import os
import re
import json
import shutil
import uuid
import io
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, json
from paddleocr import PaddleOCR,draw_ocr
import numpy as np
from PIL import Image
import cv2
import pypdfium2 as pdfium

class OcrMethod:
    ocr = None

    def __init__(self):
        if OcrMethod.ocr is None:
            OcrMethod.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def process_ocr_result(self, result, fields_input):
        detected_text_list = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                detected_text_list.append(line[1][0])
        text = ' '.join(detected_text_list)
        data = {}
        fields = json.loads(fields_input)
        if fields_input:
            for field in fields:
                key = field.get("key")
                pattern = field.get("pattern")
                repeatable = field.get("repeatable", True)
                table = field.get("table")

                if table == False:
                    if pattern:
                        matches = re.findall(pattern, text, flags=re.IGNORECASE)
                        if matches and table == False:
                            if repeatable and table == False:
                                data[key] = list(set(matches))
                            else:
                                data[key] = matches[0]
                elif table == True:
                    table_pattern = json.loads(fields_input)
                    matched_data = {}
                    for f in table_pattern:
                        key1 = f.get("key")
                        pattern1 = f.get("pattern")
                        repeatable1 = f.get("repeatable", True)
                        table1 = f.get("table")

                        if table1 == True:
                            if pattern1:
                                matches1 = re.findall(pattern1, text, flags=re.IGNORECASE)
                                if matches1 and table1 == True:
                                    if key1 not in matched_data:
                                        matched_data[key1] = []

                                    if repeatable1:
                                        matched_data[key1].extend(matches1)
                                    else:
                                        matched_data[key1].append(matches1[0])

                    output_data = []
                    keys = list(matched_data.keys())
                    if keys:
                        max_entries = max(len(matched_data[key]) for key in keys)
                        for i in range(max_entries):
                            entry = {}
                            for key in keys:
                                if i < len(matched_data[key]):
                                    entry[key] = matched_data[key][i]
                                else:
                                    entry[key] = None
                            output_data.append(entry)
                        data['table_data'] = output_data
                    else:
                        data['table_data'] = "No matching data found for table patterns."

                    return data

    def process_image(self, image_data):
        result = OcrMethod.ocr.ocr(image_data)
        return  result

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
            result = OcrMethod.ocr.ocr(image)
            os.remove(image_path)
        pdf.close()
        return result

    def converterfile(self, file, fields_input):
        if file is None:
            return jsonify({"error": "No file selected"})
        file_end = file.filename.endswith(".pdf")

        if not file_end:
            image_data = file.read()
            result = self.process_image(image_data)
            if result is None or len(result) == 0:
                return jsonify({"message": "OCR did not detect any text in the image"})
        else:
            pdf_data = file.read()
            result = self.process_pdf(pdf_data)

        data = self.process_ocr_result(result, fields_input)

        if data:
            return jsonify(data)
        else:
            return jsonify({"error": "No matching data found"})