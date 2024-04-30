from flask import Flask, request, jsonify, send_file
import os
import re
import tempfile
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from werkzeug.utils import secure_filename
import zipfile

def generate_junit_code(files):
    try:
        model = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                               model_kwargs={"temperature": 0.5, "seed": 42,
                                             'repetition_penalty': 1.1,
                                             "max_new_tokens": 5000,
                                             "max_length": 4000})
        template = """
        You are an intelligent chatbot. Help the following question with brilliant answers.
        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=model)
        if len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400
        contains_java_code = any(file.filename.endswith(('.java')) for file in files)
        contains_js_code = any(file.filename.endswith('.ts') for file in files)
        if (contains_js_code and contains_java_code):
            return ({"messege":"please upload the either java or js files. can't support combine all."})
        if contains_java_code:
            temp_dir = tempfile.mkdtemp()
            output_file_paths = []
            response_files = []

            for file in files:
                if file.filename == '':
                    return {"error": 'No selected file'}
                filename = secure_filename(file.filename)
                output_file_path = os.path.join(temp_dir, filename)
                file.save(output_file_path)
                output_file_paths.append(output_file_path)

            zip_file_path = os.path.join(tempfile.gettempdir(), 'responses.zip')
            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                for file_path in output_file_paths:
                    with open(file_path, "r") as java_file:
                        content = java_file.read()
                        result = content + "\n\n" + "Generate JUnit5 test cases for the following program"
                        response = llm_chain.run(result)
                    extracted_java_code = extract_java_code(response)
                    response_file_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}")
                    with open(response_file_path, "w") as response_file:
                        response_file.write(extracted_java_code)
                    zipf.write(response_file_path, os.path.basename(response_file_path))
                    response_files.append(response_file_path)
            for response_file in response_files:
                os.remove(response_file)
            return send_file(zip_file_path, as_attachment=True, download_name='Junit_test_code.zip')
        if contains_js_code:
            temp_dir = tempfile.mkdtemp()
            output_file_paths = []
            response_files = []
            for file in files:
                if file.filename == '':
                    return {"error": 'No selected file'}
                filename = secure_filename(file.filename)
                output_file_path = os.path.join(temp_dir, filename)
                file.save(output_file_path)
                output_file_paths.append(output_file_path)
            zip_file_path = os.path.join(tempfile.gettempdir(), 'responses.zip')
            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                for file_path in output_file_paths:
                    with open(file_path, "r") as java_file:
                        content = java_file.read()
                        result = content + "\n\n" + "please generate JEST test cases for the following program,without Jasmine."
                        response = llm_chain.run(result)
                    extracted_java_code = extract_js_code(response)
                    response_file_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}")
                    with open(response_file_path, "w") as response_file:
                        response_file.write(extracted_java_code)
                    zipf.write(response_file_path, os.path.basename(response_file_path))
                    response_files.append(response_file_path)
            for response_file in response_files:
                os.remove(response_file)
            return send_file(zip_file_path, as_attachment=True, download_name='Jest_test_code.zip')

    except Exception as e:
        pass

def extract_java_code(response):
    answer_index = response.find("Answer:")
    if answer_index != -1:
        answer_text = response[answer_index + len("Answer:") + 1:].strip()
        start_code = answer_text.find("```java")
        end_index = answer_text.rfind("}")
        if start_code != -1 and end_index != -1:
            extracted_json = answer_text[start_code:end_index + 1]
            match = re.search(r'```java\s+', extracted_json)
            if match:
                start_index = match.end()
                output = extracted_json[start_index:]
                return output
            else:
                return answer_text
        else:
            return answer_text
    else:
        answer_text = response
        if answer_text.find("```java"):
            code_blocks = re.findall(r'```java(.*?)```', answer_text, re.DOTALL)
            code_text = "\n".join(code_blocks)
            return code_text
        else:
            return answer_text

def extract_js_code(response):
    if response.find("```javascript") != -1 or response.find("```typescript") != -1:
        answer_index = response.find("Answer:")
        answer_text = response[answer_index + len("Answer:") + 1:].strip()
        code_blocks = re.findall(r'```(javascript|typescript)(.*?)```', answer_text, re.DOTALL)
        code_text = "\n".join([block[1] for block in code_blocks])
        return code_text
    else:
        return response