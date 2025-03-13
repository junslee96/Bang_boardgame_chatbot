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

# 파일 업로더 위젯 추가
uploaded_file = st.file_uploader("첨부파일을 선택하세요.")

if uploaded_file is not None:
    # 업로드된 파일을 JSON으로 로딩
    merged_data = json.load(uploaded_file)
    
    # 문서 생성
    documents = []
    for item in merged_data:
        if 'content' in item:
            documents.append(item['content'])
    
    # 문서 벡터화
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    chunked_documents = []
    for doc in documents:
        chunked_documents.extend(doc.split())
    X = model.encode(chunked_documents)
    
    # 세션 상태에 저장
    st.session_state.documents = documents
    st.session_state.chunked_documents = chunked_documents
    st.session_state.X = X
    
    # 업로드 완료 메시지 표시
    st.success(f"파일 '{uploaded_file.name}' 업로드가 완료되었습니다!")
    
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


def replace_terms(text):
    replace_dict = {'사람': '플레이어'}
    for key, value in replace_dict.items():
        text = re.sub(key, value, text)
    return text

# 질문 변환 기법 적용
def transform_query(query):
    transformed_query = query + " 관련 정보"
    return transformed_query
def retrieve_similar_documents(query, documents, X, top_k=3):
    try:
        model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        query_vec = model.encode([query])
        
        # 문서의 청크별 임베딩을 사용하여 유사성을 계산
        similarities = np.dot(X, query_vec.T[0]).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # 문서의 원본 청크가 아닌 전체 문서를 반환하기 위해 인덱스를 매핑
        chunk_to_doc_map = {}
        chunk_index = 0
        for doc in documents:
            chunked_doc = chunk_text(doc)
            for _ in chunked_doc:
                chunk_to_doc_map[chunk_index] = doc
                chunk_index += 1
        
        # top_k 개의 문서 인덱스를 전체 문서로 매핑
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
    
    # 문장의 중요도를 평가하여 상위 N개의 문장을 선택
    sentence_scores = []
    for sentence in relevant_sentences:
        score = len([morph for morph in mecab.morphs(sentence) if morph not in stop_words])
        sentence_scores.append((sentence, score))
    
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:5]
    return '\n'.join([sentence for sentence, _ in top_sentences])



# Streamlit 앱 시작
st.title("🤠 뱅 보드게임 챗봇")
st.write(
    "첨부파일을 기반으로 만든 간단한 생성형 챗봇입니다."
    " '뱅 보드게임에서'라는 말과 함께 질문해주세요!"
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = openai.OpenAI(api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "documents" not in st.session_state:
        st.session_state.documents = []
        st.session_state.chunked_documents = []
        st.session_state.X = []

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        modified_question = prompt
        
        try:
            retrieved_docs = retrieve_similar_documents(modified_question, st.session_state.chunked_documents, st.session_state.X)
            context = create_context(retrieved_docs)
            answer_prompt = f"컨텍스트: {context}\n\n질문: {modified_question}\n답변:"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "assistant", "content": answer_prompt}
                ],
                max_tokens=1000,
                temperature=0.2,  # 낮은 온도 설정
                top_p=0.95,  # top_p를 0에 가깝게 설정
                frequency_penalty=1.3,  # frequency_penalty 사용
                stream=False
            )
            
            answer = response.choices[0].message.content
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            with st.chat_message("assistant"):
                st.markdown(answer)
        
        except Exception as e:
            st.error(f"An error occurred while generating a response: {e}")


