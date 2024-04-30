import tempfile
import io
import re
import shutil
import textwrap
import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cohere
import warnings
warnings.filterwarnings("ignore")
from flask import jsonify
from spellchecker import SpellChecker
from paddleocr import PaddleOCR

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
import atexit
from transformers import pipeline,set_seed
from sklearn.metrics import (
    accuracy_score,f1_score,recall_score,
    precision_score,confusion_matrix)

from langchain_community.document_loaders import TextLoader,Docx2txtLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ocr.image_to_pattern import HugggingfaceModel,Huggingface_mistral
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.llms import CTransformers,HuggingFaceHub
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from docx import Document
import docx2txt

class ConversationalChatBot:
    llm = None
    def __init__(self):
        set_seed(42)
        torch.manual_seed(42)

        if ConversationalChatBot.llm is None:
            config = {'max_new_tokens': 1600,
                      'repetition_penalty': 1.1,
                      'context_length': 6000}

            model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "model", "Mistral-7B-Instruct-v0.2-GGUF",
                             "mistral-7b-instruct-v0.2.Q2_K.gguf")
            )
            ConversationalChatBot.llm = CTransformers(
                model= model_path,
                model_type='mistral',
                config=config,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
    def filter_answer(self,input_query,retriver):
        relevent_text = retriver.get_relevant_documents(input_query)
        page_content_list = []
        for document in relevent_text:
            page_content = document.page_content
            page_content_list.append(page_content)
        page_content = " ".join(page_content_list[0:])
        template = """
        You are an intelligent chatbot. Help the following question with brilliant answers.
        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=ConversationalChatBot.llm)
        result = page_content + "\n\n\n" +"Please provide an answer to the following question using the text data above. Make sure the answer is relevant to the provided text."+"\n"+input_query
        response = llm_chain.run(result)
        answer_index = response.find("Answer:")
        if answer_index != -1:
            answer_text = response[answer_index + len("Answer:") + 1:].strip()
            return ({"result":answer_text.strip()})
        else:
            return ({"result":response.strip()})

    def spell_check_and_correct(self,input_text):
        spell = SpellChecker()
        words = input_text.split()
        corrected_text = []
        for word in words:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word if corrected_word is not None else word)
        return ' '.join(corrected_text)

    def remove_punc(self,x):
        punctuation = '!"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~'
        text = "".join([char for char in x if char not in punctuation])
        return text

    def check_the_score(self,page_content,input_query):
        texts = [page_content,input_query]
        vector = TfidfVectorizer()
        vector1 = CountVectorizer()
        count_matrix = vector.fit_transform(texts)
        match = cosine_similarity(count_matrix)[0][1]
        match = round(match * 100, 2)
        return str(match)

    def yorogpt(self,input_query):
        text_prompt = "<s>[INST] What is your favourite condiment? [/INST]"
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        "[INST] Do you have mayonnaise recipes? [/INST]"
        template = """
        You are an intelligent chatbot. Help the following question with brilliant answers.
        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=ConversationalChatBot.llm)
        response = llm_chain.run(input_query)
        answer_index = response.find("Answer:")
        if answer_index != -1:
            answer_text = response[answer_index + len("Answer:") + 1:].strip()
            return ({"result":answer_text.strip()})
        else:
            return ({"result":response.strip()})

    def yorochatbot(self,input_query):
        data = pd.read_csv("h2oGPT/datasets.csv", encoding="iso-8859-1")
        df = pd.DataFrame(data)
        df.dropna(inplace=True)
        df["Question"] = df["Question"].str.lower()
        df["Question"] = df['Question'].apply(lambda x: self.remove_punc(x))
        x, y = (df.iloc[:, 0], df.iloc[:, 1])
        count = TfidfVectorizer(stop_words="english")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=3)
        x_train = count.fit_transform(x_train)
        x_test = count.transform(x_test)
        log = LogisticRegression(max_iter=100, penalty="l2", solver="liblinear", class_weight="balanced")
        log.fit(x_train, y_train)
        test_accuracy = log.score(x_test, y_test)
        train_accuracy = log.score(x_train, y_train)
        corrected_query = self.spell_check_and_correct(input_query)
        message = corrected_query.lower()
        msginput = count.transform([message])
        predict = log.predict(msginput)
        output_text = predict[0].strip("[]'")
        result = {"result": output_text}
        return jsonify(result)

    def content_evaluation(self,page_content,input_query):
        template = """
        {question}
        Answer: """
        prompt = PromptTemplate(template=template, input_variables=["prompt"])
        llm_chain = LLMChain(prompt=prompt, llm=ConversationalChatBot.llm)
        similarity_score = self.check_the_score(page_content, input_query)
        score = "DATA 1 AND DATA 2 SIMILARITY MATCH SCORE:" + "\n\n" + f"similarity match score or ATS score = {similarity_score}"
        result =score+"\n\n"+page_content + "\n\n\n" + "Please provide an answer to the following question using the text data above. Make sure the answer is relevant to the provided text." +"\n"+input_query
        response = llm_chain.run(result)
        answer_index = response.find("Answer:")
        if answer_index != -1:
            end_index = response.find("\nAnswer:", answer_index + len("Answer:") + 1)
            if end_index != -1:
                answer_text = response[answer_index + len("Answer:") + 1:end_index].strip()
            else:
                answer_text = response[answer_index + len("Answer:") + 1:].strip()
            return ({"result": answer_text.strip()})
        else:
            return ({"result": response.strip()})


    def documentchatbot(self, files, input_query):
        text_prompt = "<s>[INST] What is your favourite condiment? [/INST]"
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        "[INST] Do you have mayonnaise recipes? [/INST]"
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        if len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400
        contains_image = any(file.filename.endswith(('.jpg', '.jpeg', '.png','.webp')) for file in files)
        contains_pdf = any(file.filename.endswith('.pdf') for file in files)
        contains_text = any(file.filename.endswith('.txt') for file in files)
        contains_word = any(file.filename.endswith(('.docx','.doc')) for file in files)
        if (contains_image and contains_pdf) or (contains_image and contains_word) or (
                contains_pdf and contains_word) or (contains_image and contains_text) or (
                contains_pdf and contains_text) or (contains_word and contains_text):
            return jsonify(
                {'error': 'Please upload either PDF or Image or text or word, not support combining all.'}), 400

        if contains_image:
            ocr = PaddleOCR(use_angle_cls=True, lang="en")
            detected_text = ""
            for idx, file in enumerate(files, start=1):
                image_data = file.read()
                result = ocr.ocr(image_data)
                detected_text += f"Image{idx} text:\n"
                for res in result:
                    for line in res:
                        detected_text += line[1][0] + " "
                detected_text += "\n\n"
            return detected_text

        elif contains_pdf:
            all_text = ''
            for file in files:
                filename = secure_filename(file.filename)
                file_path = os.path.join(tempfile.mkdtemp(), filename)
                file.save(file_path)
                with open(file_path, 'rb') as pdf_file:
                    text = extract_text(pdf_file)
                    all_text += text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=10)
            chunks = text_splitter.split_text(all_text)
            vector_store = FAISS.from_texts(chunks, embeddings)
            docs = vector_store.similarity_search(input_query, k=3)
            page_content_list = []
            for document in docs:
                page_content = document.page_content
                page_content_list.append(page_content)
            page_content = " ".join(page_content_list[0:])
            answer = self.content_evaluation(page_content,input_query)
            return answer

        elif contains_word:
            all_texts = ''
            for file in files:
                if file:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(tempfile.mkdtemp(), filename)
                    file.save(file_path)
                    with open(file_path, 'rb') as word_file:
                        doc = docx2txt.process(file_path)
                        all_texts += doc + "\n"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=10)
            chunks = text_splitter.split_text(all_texts)
            vector_store = FAISS.from_texts(chunks, embeddings)
            docs = vector_store.similarity_search(input_query, k=2)
            page_content_list = []
            for document in docs:
                page_content = document.page_content
                page_content_list.append(page_content)
            page_content = " ".join(page_content_list[0:])
            answer = self.content_evaluation(page_content,input_query)
            return answer

        elif contains_text:
            all_texts = ''
            for file in files:
                filename = secure_filename(file.filename)
                file_path = os.path.join(tempfile.mkdtemp(), filename)
                file.save(file_path)
                with open(file_path, 'rb') as text_file:
                    file_contents = text_file.read()
                    all_texts += file_contents.decode('utf-8')
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_text(all_texts)
            vector_store = FAISS.from_texts(chunks, embeddings)
            retriver = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
            output_answer = self.filter_answer(input_query,retriver)
            return output_answer
        else:
            return jsonify({'error': 'Invalid file format'}), 400