import streamlit as st
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import requests

# 깃허브 저장소 정보
owner = "junslee96"
repo = "Bang_boardgame_chatbot"
file_path = "prompt_data"

# 룰북 파일들 읽기
rulebook_contents = []
for i in range(1, 12):  # 1부터 11까지
    url = f"https://github.com/{owner}/{repo}/tree/main/{file_path}/뱅!_룰북_{i}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        rulebook_contents.append(response.text)

# QA 데이터 읽기
qa_data_url = f"https://github.com/{owner}/{repo}/tree/main/{file_path}/qa종합_최종_modified_수정.xlsx"
response = requests.get(qa_data_url)
if response.status_code == 200:
    # 엑셀 파일을 읽기 위해 임시 파일로 저장
    with open("temp.xlsx", "wb") as file:
        file.write(response.content)
    qa_df = pd.read_excel("temp.xlsx", engine='openpyxl')
else:
    print("QA 데이터를 읽을 수 없습니다.")
    qa_df = None

# 기존 documents 리스트에 룰북과 QA 데이터 추가
documents = []

# 룰북 내용 추가
documents.extend(rulebook_contents)

# QA 데이터 추가
if qa_df is not None:
    for _, row in qa_df.iterrows():
        documents.append(f"질문: {row['질문']} 답변: {row['답변']}")

# 청크 크기 조정 및 청크 생성(학습 데이터 잘 읽히기)
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunked_documents = []
for doc in documents:
    chunked_documents.extend(chunk_text(doc))
documents = chunked_documents

# 벡터화(문장 조리있게 정리)
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
X = model.encode(documents)

# 벡터화된 데이터의 차원 확인 (디버깅용)
print(f"X shape: {X.shape}")

# 유사 문서 검색 함수 개선(질문 문서 1개 -> 여러 개)
def retrieve_similar_documents(query, top_k=3):
    try:
        query_vec = model.encode([query])
        print(f"query_vec shape: {query_vec.shape}")  # 디버깅용 출력
        
        # 차원 불일치 해결을 위해 query_vec.T[0] 사용
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
            retrieved_docs = retrieve_similar_documents(modified_question)

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
