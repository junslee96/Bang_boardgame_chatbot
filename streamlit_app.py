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

        # 두 데이터셋을 통합하여 문서 생성
        documents = []
        for item in merged_data:
            if 'content' in item:
                documents.append(item['content'])

        if qa_df is not None:
            for _, row in qa_df.iterrows():
                # 질문과 답변을 하나의 문서로 통합
                documents.append(f"질문: {row['질문']}\n답변: {row['답변']}")

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

        chunk_to_doc_map = {}
        chunk_index = 0
        for doc in documents:
            chunked_doc = chunk_text(doc)
            for _ in chunked_doc:
                chunk_to_doc_map[chunk_index] = doc
                chunk_index += 1

        top_docs = []
        for idx in top_indices:
            if idx in chunk_to_doc_map:
                top_docs.append(chunk_to_doc_map[idx])

        return top_docs

    except Exception as e:
        print(f"Error in retrieve_similar_documents: {e}")
        return []


def create_context(retrieved_docs):
    mecab = Mecab()
    stop_words = set(['를', '을', '는', '이', '가', '에', '와', '과', '으로', '에서', '까지'])

    relevant_sentences = []
    for doc in retrieved_docs:
        sentences = doc.split('. ')
        for sentence in sentences:
            morphs = mecab.morphs(sentence)
            if len(morphs) > 10 and not any(morph in stop_words for morph in morphs):
                relevant_sentences.append(sentence)

    sentence_scores = []
    for sentence in relevant_sentences:
        score = len([morph for morph in mecab.morphs(sentence) if morph not in stop_words])
        sentence_scores.append((sentence, score))

    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:5]
    return '\n'.join([sentence for sentence, _ in top_sentences])


def generate_response(query, conversation_history, persona_profile):
    if "chunked_documents" not in st.session_state or "X" not in st.session_state:
        return "현재 데이터가 초기화되지 않았습니다. 데이터를 다시 로드해주세요."

    retrieved_docs = retrieve_similar_documents(query, st.session_state["chunked_documents"], st.session_state["X"])
    context = create_context(retrieved_docs)
    answer_prompt = f"{persona_profile}\n\n{conversation_history}\n\n질문: {query}\n답변:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "assistant", "content": answer_prompt}
        ],
        max_tokens=1000,
        temperature=0.2,
        top_p=0.01,
        frequency_penalty=1.2,
        stream=False
    )

    answer = response.choices[0].message.content
    return answer


def replace_terms(text):
    replace_dict = {'사람': '플레이어'}
    for key, value in replace_dict.items():
        text = re.sub(key, value, text)
    return text


def transform_query(query):
    transformed_query = query + " 관련 정보"
    return transformed_query


st.title("🤠 뱅 보드게임 챗봇")
st.write(
    "OpenAI의 gpt-4o-mini 모델을 사용해서 만든 간단한 생성형 챗봇입니다."
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = openai.OpenAI(api_key=openai_api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "documents" not in st.session_state:
    st.session_state["documents"] = load_data() or []

if "chunked_documents" not in st.session_state or "X" not in st.session_state:
    if st.session_state["documents"]:
        st.session_state["chunked_documents"], st.session_state["X"] = vectorize_documents(st.session_state["documents"])
    else:
        st.session_state["chunked_documents"] = []
        st.session_state["X"] = []


persona_profile = """
이름: 뱅! 보드게임 가이드
나이: 25-40세
관심사: 서부 시대, 보드게임, 전략, 협동
역할: 보안관, 부관, 무법자, 배신자
목표: 게임에서 승리하기 위해 역할에 맞는 목표를 달성
특성: 전략적이고, 상황에 맞게 적응하며, 때로는 위험을 감수하기도 함

예시 대화:
"안녕하세요! 뱅! 보드게임을 시작하기 전에..."
"""

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    modified_question = replace_terms(prompt)

    try:
        conversation_history = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
        
        answer = generate_response(modified_question, conversation_history, persona_profile)
        
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        
        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as e:
        st.error(f"An error occurred while generating a response: {e}")
