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

# 깃허브 저장소 정보
owner = "junslee96"
repo = "Bang_boardgame_chatbot"
file_path = "prompt_data"

# 룰북 파일들 읽기
merged_data_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}/merged_data.json"

# QA 데이터 읽기
output_data_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}/output_data.json"

# json 파일 읽기 및 데이터 준비
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

        qa_data = []
        if qa_df is not None:
            for _, row in qa_df.iterrows():
                qa_data.append({"질문": row['질문'], "답변": row['답변']})

        return documents, qa_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Load data only once when the app starts
if "documents" not in st.session_state or "qa_data" not in st.session_state:
    st.session_state.documents, st.session_state.qa_data = load_data()
    st.session_state.chunked_documents, st.session_state.X = vectorize_documents(st.session_state.documents)

# Create a chat input field to allow the user to enter a message.
if prompt := st.chat_input("What is up?"):
    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 질문에 '사람'을 '플레이어'로 대체
    modified_question = replace_terms(prompt)

    try:
        # QA 데이터에서 관련된 답변 검색
        for qa in st.session_state.qa_data:
            if modified_question in qa['질문']:
                answer = qa['답변']
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                break
        else:
            # QA 데이터에서 관련된 답변을 찾지 못한 경우, OpenAI API 사용
            retrieved_docs = retrieve_similar_documents(modified_question, st.session_state.chunked_documents, st.session_state.X)
            context = "\n".join(retrieved_docs)
            answer_prompt = f"컨텍스트: {context}\n\n질문: {modified_question}\n답변:"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "assistant", "content": answer_prompt}
                ],
                max_tokens=200,
                stream=False
            )
            answer = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        
    except Exception as e:
        st.error(f"An error occurred while generating a response: {e}")
