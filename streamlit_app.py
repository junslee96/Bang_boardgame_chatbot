import streamlit as st
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import requests
import json
import nest_asyncio
import asyncio
from ekonlpy.tag import Mecab  # eKoNLPy 사용

nest_asyncio.apply()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 깃허브 저장소 정보
owner = "junslee96"
repo = "Bang_boardgame_chatbot"
file_path = "prompt_data"

merged_data_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}/merged_data.json"
output_data_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}/output_data.json"

def load_data():
    try:
        response_merged = requests.get(merged_data_url)
        if response_merged.status_code == 200:
            merged_data = response_merged.json()
        else:
            print(f"Failed to read merged data. Status code: {response_merged.status_code}")
            return None

        response_qa = requests.get(output_data_url)
        if response_qa.status_code == 200:
            qa_data = response_qa.json()
            qa_df = pd.DataFrame(qa_data)
        else:
            print(f"Failed to read QA data. Status code: {response_qa.status_code}")
            qa_df = None

        documents = []
        for item in merged_data:
            if 'content' in item:
                documents.append(item['content'])

        if qa_df is not None:
            for _, row in qa_df.iterrows():
                documents.append(f"질문: {row['질문']} 답변: {row['답변']}")
        return documents
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def vectorize_documents(documents):
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    chunked_documents = []
    for doc in documents:
        chunked_documents.extend(chunk_text(doc))
    X = model.encode(chunked_documents)
    return chunked_documents, X

def retrieve_similar_documents(query, documents, X, top_k=3):
    try:
        model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        query_vec = model.encode([query])
        similarities = np.dot(X, query_vec.T[0]).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [documents[i] for i in top_indices]
    except Exception as e:
        print(f"Error in retrieve_similar_documents: {e}")
        return []

def replace_terms(text):
    replace_dict = {'사람': '플레이어'}
    for key, value in replace_dict.items():
        text = re.sub(key, value, text)
    return text

def create_context(retrieved_docs):
    mecab = Mecab()  # eKoNLPy의 Mecab 형태소 분석기 사용
    stop_words = set(['를', '을', '는', '이', '가', '에', '와', '과', '으로', '에서', '까지'])
    
    relevant_sentences = []
    for doc in retrieved_docs:
        sentences = doc.split('. ')  # 간단한 문장 분리 (Mecab에 문장 분리 기능 없음)
        for sentence in sentences:
            morphs = mecab.morphs(sentence)
            if len(morphs) > 10 and not any(morph in stop_words for morph in morphs):
                relevant_sentences.append(sentence)
    
    return '\n'.join(relevant_sentences)

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = openai.OpenAI(api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "documents" not in st.session_state:
        st.session_state.documents = load_data()
        st.session_state.chunked_documents, st.session_state.X = vectorize_documents(st.session_state.documents)

    if prompt := st.chat_input("What is up?"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
          st.markdown(prompt)

      modified_question = replace_terms(prompt)

      try:
          retrieved_docs = retrieve_similar_documents(modified_question, st.session_state.chunked_documents, st.session_state.X)
          context = create_context(retrieved_docs)
          answer_prompt = f"컨텍스트: {context}\n\n질문: {modified_question}\n답변:"

          response = client.chat.completions.create(
              model="gpt-4o-mini",
              messages=[
                  {"role": "assistant", "content": answer_prompt}
              ],
              max_tokens=250,
              stream=False
          )

          answer = response.choices[0].message.content
          st.session_state.messages.append({"role": "assistant", "content": answer})
          with st.chat_message("assistant"):
              st.markdown(answer)
      
      except Exception as e:
          st.error(f"An error occurred while generating a response: {e}")
