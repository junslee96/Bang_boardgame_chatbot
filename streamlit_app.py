def generate_response(query):
    retrieved_docs = retrieve_similar_documents(query, st.session_state.chunked_documents, st.session_state.X)
    context = create_context(retrieved_docs)
    answer_prompt = f"컨텍스트: {context}\n\n질문: {query}\n답변:"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "assistant", "content": answer_prompt}
        ],
        max_tokens=1000,
        temperature=0.2,  # 낮은 온도 설정
        top_p=0.01,  # top_p를 0에 가깝게 설정
        frequency_penalty=1.2,  # frequency_penalty 사용
        stream=False
    )
    
    answer = response.choices[0].message.content
    
    # merged_data의 문맥을 반영하여 후처리
    processed_answer = f"질문: {query}\n답변: {answer}"
    
    # merged_data의 형식이나 구조를 따르도록 수정
    # 예: 특정 키워드 삽입, 문장 구조 조정 등
    
    return processed_answer
