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

        if qa_df is not None:
            for _, row in qa_df.iterrows():
                documents.append(f"질문: {row['질문']} 답변: {row['답변']}")
        return documents
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# 청크 크기 조정 및 청크 생성
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# 벡터화(문장 조리있게 정리)
def vectorize_documents(documents):
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    chunked_documents = []
    for doc in documents:
        chunked_documents.extend(chunk_text(doc))
    X = model.encode(chunked_documents)
    return chunked_documents, X

# 유사 문서 검색 함수
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

# '사람'을 '플레이어'로 대체하는 함수
def replace_terms(text):
    replace_dict = {'사람': '플레이어'}
    for key, value in replace_dict.items():
        text = re.sub(key, value, text)
    return text

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's gpt-4o-mini model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = openai.OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load data only once when the app starts
    if "documents" not in st.session_state:
        st.session_state.documents = load_data()
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
            # 유사한 문서 검색
            retrieved_docs = retrieve_similar_documents(modified_question, st.session_state.chunked_documents, st.session_state.X)

            # 컨텍스트 생성
            context = "\n".join(retrieved_docs)
            answer_prompt = f"컨텍스트: {context}\n\n질문: {modified_question}\n답변:"

            # OpenAI API를 사용하여 답변 생성
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "assistant", "content": answer_prompt}
                ],
                max_tokens=200,
                stream=False
            )

            # 답변 저장 및 표시
            answer = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        
        except Exception as e:
            st.error(f"An error occurred while generating a response: {e}")
