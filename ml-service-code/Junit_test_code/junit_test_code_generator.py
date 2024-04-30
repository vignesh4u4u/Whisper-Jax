from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
import os
import re
import tempfile
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from werkzeug.utils import secure_filename

def generate_junit_code(files):
    config = {'max_new_tokens': 1600,
              'repetition_penalty': 1.1,
              'seed': 42,
              'context_length': 6000}

    model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', 'model', 'Llama-2-7B-Chat-GGUF', 'llama-2-7b-chat.Q2_K.gguf'
    ))
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "model", "Mistral-7B-Instruct-v0.2-GGUF",
                     "mistral-7b-instruct-v0.2.Q2_K.gguf")
    )
    try:
        model1 = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                               model_kwargs={"temperature": 0.5, "seed": 42,
                                             'repetition_penalty': 1.1,
                                             "max_new_tokens": 5000,
                                             "max_length": 4000})
        template = """
        You are an intelligent chatbot. Help the following question with brilliant answers.
        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=model1)
        if len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400
        contains_java_code = any(file.filename.endswith(('.java')) for file in files)
        contains_js_code = any(file.filename.endswith('.ts') for file in files)
        if (contains_js_code and contains_java_code):
            return ({"messege":"please upload the either java or js files. can't support combine all."})
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
            with open(output_file_path, 'r') as text_file:
                file_contents = text_file.read()
        if contains_java_code:
            result = file_contents + "\n\n\n" + "Please generate JUnit test cases code for this program."
            response = llm_chain.run(result)
            answer = extract_java_code(response)
            return answer
        if contains_js_code:
            result = file_contents + "\n\n\n" + "please generate JEST test cases for the following program,without Jasmine."
            response = llm_chain.run(result)
            answer = extract_js_code(response)
            return answer

    except Exception as e:
        return ({"error":str(e)})

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