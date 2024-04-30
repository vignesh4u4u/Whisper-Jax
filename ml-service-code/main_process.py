import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
from PdfTextExtraction import text_extract_pdf
from ocr import image_to_json
from ocr import image_text_coordinates
from ocr import image_text_drawbox
from ocr import image_to_json_ai
from ocr import image_to_key_value
from h2oGPT import automatic_image_to_json
from h2oGPT import conversational_chatbot
from h2oGPT import content_generation
from h2oGPT import mistralAI
from PdfTextExtraction import huggingfaceai
from LocationFinder import ipaddress_to_locationfinder
from Junit_test_code import automatic_junit_code_generator
from Junit_test_code import junit_test_code_generator
from audio_video_speech_recognition import video_audio_to_text_converter
from audio_video_speech_recognition import assemblyai
from audio_video_speech_recognition import voice_to_speaker_text
from llama3_test import llama3_conversional_chat

CONTEXT_PATH = "/ml-service"

flask_app = Flask(__name__)
DEBUG = False

@flask_app.route(CONTEXT_PATH + "/health/v1/ping", methods=["GET"])
def health_check():
    return "pong"

@flask_app.route(CONTEXT_PATH + "/text-extraction/v1/extract", methods=["POST"])
def pdf_text_extraction():
    extractor = text_extract_pdf.PDFTextExtrator()
    print("inside the extraction")
    file = request.files["files"]
    selected_options = {}
    if ('extractOptions' in request.form):
        selected_options = request.form["extractOptions"]
    print (selected_options)
    response_data = extractor.extract_text(file, selected_options)
    return response_data

@flask_app.route(CONTEXT_PATH + "/image-to-json/v1/extract", methods=["POST"])
def ocr_data_extraction():
    converter_json = image_to_json.OcrMethod()
    file = request.files["file"]
    fields_input = request.form["fields"]
    return converter_json.converterfile(file, fields_input)

@flask_app.route(CONTEXT_PATH + "/image-text-coordinates/v1/extract", methods=["POST"])
def image_text_coordinate():
    converter_coordinates = image_text_coordinates.OcrMethodCoordinates()
    file = request.files["file"]
    return converter_coordinates.coordinates(file)

@flask_app.route(CONTEXT_PATH + "/image-text-boundingbox/v1/extract", methods=["POST"])
def image_text_boundingbox():
    converter_box = image_text_drawbox.OcrMethodDrawbox()
    file = request.files["file"]
    return converter_box.boundingbox(file)

@flask_app.route(CONTEXT_PATH + "/automatic-images-to-json/v1/extract", methods=["POST"])
def image_text_keyvaluepair():
    change_key_name = automatic_image_to_json.KeyNameMapping()
    try:
        if "file" not in request.files:
            return "No file part in the request"
        image_file = request.files["file"]
        if "fields" in request.form and request.form["fields"].strip() != "" and request.form["fields"] is not None:
            key_mapping = request.form["fields"]
            final_answer = change_key_name.convert(image_file, key_mapping)
            return final_answer
        else:
            final_answer = change_key_name.without_replacekeys(image_file)
            return final_answer
    except Exception as e:
        return jsonify({"error":str(e)}),500

@flask_app.route(CONTEXT_PATH + "/content-generation/v1/generator", methods=["POST"])
def content_generator():
    converter_chat = content_generation.ConversationalChatBot()
    try:
        if "file" in request.files and "input_prompt" in request.form:
            files = request.files.getlist("file")
            input_query = request.form["input_prompt"]
            return converter_chat.documentchatbot(files,input_query)
        elif "input_prompt" in request.form:
            input_query = request.form["input_prompt"]
            return converter_chat.yorogpt(input_query)
        else:
            return "Error: Missing required parameters 'file' or 'input_prompt'.", 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route(CONTEXT_PATH + "/text-generation/v1/generator", methods=["POST"])
def textgeneration():
    converter_chat = mistralAI.ConversationalChatBot()
    try:
        if "file" in request.files and "input_prompt" in request.form:
            files = request.files.getlist("file")
            input_query = request.form["input_prompt"]
            return converter_chat.documentchatbot(files,input_query)
        elif "input_prompt" in request.form:
            input_query = request.form["input_prompt"]
            return converter_chat.yorogpt(input_query)
        else:
            return "Error: Missing required parameters 'file' or 'input_prompt'.", 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route(CONTEXT_PATH + "/llama3-content-generation/v1/generator", methods=["POST"])
def llama3_content_generator():
    converter_chat = llama3_conversional_chat.ConversationalChatBot()
    try:
        if "file" in request.files and "input_prompt" in request.form:
            files = request.files.getlist("file")
            input_query = request.form["input_prompt"]
            return converter_chat.documentchatbot(files,input_query)
        elif "input_prompt" in request.form:
            input_query = request.form["input_prompt"]
            return converter_chat.yorogpt(input_query)
        else:
            return "Error: Missing required parameters 'file' or 'input_prompt'.", 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route(CONTEXT_PATH + "/ocr-automatic-images-to-json/v1/extract", methods=["POST"])
def ocr_image_text_keyvaluepair():
    change_key_name = image_to_json_ai.KeyNameMapping()
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        image_file = request.files["file"]
        if "fields" in request.form and request.form["fields"].strip() != "" and request.form["fields"] is not None:
            key_mapping = request.form["fields"]
            final_answer = change_key_name.convert(image_file, key_mapping)
            return final_answer
        else:
            final_answer = change_key_name.without_replacekeys(image_file)
            return final_answer
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route(CONTEXT_PATH + "/llama-automatic-images-to-json/v1/extract", methods=["POST"])
def llama_image_text_keyvaluepair():
    change_key_name = image_to_key_value.KeyNameMapping()
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        image_file = request.files["file"]
        if "fields" in request.form and request.form["fields"].strip() != "" and request.form["fields"] is not None:
            key_mapping = request.form["fields"]
            final_answer = change_key_name.convert(image_file, key_mapping)
            return final_answer
        else:
            final_answer = change_key_name.without_replacekeys(image_file)
            return final_answer
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route(CONTEXT_PATH + "/conversational-chatbot/v1/chat", methods=["POST"])
def conversational():
    converter_chat = huggingfaceai.ConversationalChatBot()
    try:
        if "file" in request.files and "input_prompt" in request.form:
            files = request.files.getlist("file")
            input_query = request.form["input_prompt"]
            return converter_chat.documentchatbot(files, input_query)
        elif "input_prompt" in request.form:
            input_query = request.form["input_prompt"]
            return converter_chat.yorogpt(input_query)
        else:
            return "Error: Missing required parameters 'file' or 'input_prompt'.", 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route(CONTEXT_PATH + "/location-finder/v1/location/<ip_address>", methods=["GET"])
def get_location(ip_address):
    location_data = ipaddress_to_locationfinder.find_location(ip_address)
    return jsonify(location_data)

@flask_app.route(CONTEXT_PATH + "/automatic-junit-test-code/v1/generator", methods=["POST"])
def automatic_junit_test_code():
    try:
        if "file" not in request.files:
            return {"message":"please select the file"}
        else:
            files = request.files.getlist("file")
            generate = automatic_junit_code_generator.generate_junit_code(files)
            return generate
    except Exception as e:
        return {"error":str(e)}


@flask_app.route(CONTEXT_PATH + "/generate-junit-test-code/v1/generator", methods=["POST"])
def junit_test_code():
    try:
        if "file" not in request.files:
            return {"message":"please select the file"}
        else:
            files = request.files.getlist("file")
            generate = junit_test_code_generator.generate_junit_code(files)
            return generate
    except Exception as e:
        return {"error":str(e)}

@flask_app.route(CONTEXT_PATH + "/llama3-ocr-test/v1/extract", methods=["POST"])
def llama3_ocr_result():
    change_key_name = llama3_instruct_ocr.KeyNameMapping()
    try:
        if "file" not in request.files:
            return "No file part in the request"
        image_file = request.files["file"]
        if "fields" in request.form and request.form["fields"].strip() != "" and request.form["fields"] is not None:
            key_mapping = request.form["fields"]
            final_answer = change_key_name.convert(image_file, key_mapping)
            return final_answer
        else:
            final_answer = change_key_name.without_replacekeys(image_file)
            return final_answer
    except Exception as e:
        return jsonify({"error":str(e)}),500


@flask_app.route(CONTEXT_PATH + "/binary_files_to_text/v1/generator", methods=["POST"])
def extract_text_in_binary_files():
    try:
        if "file" not in request.files:
            return {"message":"please select the file"}
        else:
            input_file = request.files["file"]
            answer = video_audio_to_text_converter.extract_text_binary_files(input_file)
            return answer
    except Exception as e:
        return {"error":str(e)}

@flask_app.route(CONTEXT_PATH + "/separate_voices_to_text/v1/generator", methods=["POST"])
def separate_voices_text():
    try:
        if "file" not in request.files:
            return {"message":"please select the file"}
        else:
            file = request.files["file"]
            answer = voice_to_speaker_text.convert_audio_to_text(file)
            return answer
    except Exception as e:
        return {"error":str(e)}

if __name__ == "__main__":
    print("Starting the server on port 8080")
    #flask_app.run(debug=False, host="0.0.0.0", port=8080)
    http_server = WSGIServer(('0.0.0.0', 8080), flask_app)
    print('Server running on http://0.0.0.0:8080')
    http_server.serve_forever()