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

nest_asyncio.apply()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# API 키 설정
st.title("OpenAI API 키 입력")
api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")

if not api_key:
    st.info("API 키를 입력해주세요.")
else:
    client = openai.OpenAI(api_key=api_key)

    # GPT-4o-mini와 대화하는 함수
    def chat_with_gpt4omini(prompt, max_tokens=100):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"

    # 룰북 파일 읽기
    def read_rulebook(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    # 룰북 파일들 읽기
    rulebook_contents = []
    for i in range(1, 12):  # 1부터 11까지
        file_path = f'/content/뱅!_룰북_{i}.txt'
        try:
            content = read_rulebook(file_path)
            rulebook_contents.append(content)
        except FileNotFoundError:
            print(f"File {file_path} not found.")

    # QA 데이터 읽기
    qa_data_path = '/content/qa종합_최종_modified.xlsx'
    try:
        qa_df = pd.read_excel(qa_data_path, engine='openpyxl')
    except FileNotFoundError:
        print(f"File {qa_data_path} not found.")
        qa_df = None

    # 기존 documents 리스트에 룰북과 QA 데이터 추가
    documents = []

    # 룰북 내용 추가
    documents.extend(rulebook_contents)

    # QA 데이터 추가
    if qa_df is not None:
        for _, row in qa_df.iterrows():
            documents.append(f"질문: {row['질문']} 답변: {row['답변']}")

    # 청크 크기 조정 및 청크 생성
    def chunk_text(text, chunk_size=200):
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    chunked_documents = []
    for doc in documents:
        chunked_documents.extend(chunk_text(doc))
    documents = chunked_documents

    # 벡터화
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    X = model.encode(documents)

    # 유사 문서 검색 함수 개선
    def retrieve_similar_documents(query, top_k=3):
        query_vec = model.encode([query])
        similarities = np.dot(X, query_vec.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [documents[i] for i in top_indices]

    # RAG 답변 생성 함수 개선
    def rag_answer(question):
        retrieved_docs = retrieve_similar_documents(question)
        context = "\n".join(retrieved_docs)
        answer_prompt = f"컨텍스트: {context}\n\n질문: {question}\n답변:"
        answer = chat_with_gpt4omini(answer_prompt, max_tokens=200)
        return answer

    # Create a session state variable to store the chat messages.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message.
    if prompt := st.chat_input("뱅 보드게임에 대해 질문하세요: "):
        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            # RAG 답변 생성
            response = rag_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        
        except Exception as e:
            st.error(f"An error occurred while generating a response: {e}")
