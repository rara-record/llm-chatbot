"""Streamlit 앱의 진입점 모듈."""
import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

st.set_page_config(page_title="소득세 챗봇", page_icon='🤖')
st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

load_dotenv()

# Initialize chat history
if "message_list" not in st.session_state:
    st.session_state.message_list = [{"role": "ai", "content": "무엇이 궁금하신가요?"}]

# Display chat messages from history on app rerun
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept user input
if user_question := st.chat_input("소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        with st.chat_message("ai"):
            ai_response = get_ai_response(user_question)
            ai_message = st.write_stream(ai_response)
    st.session_state.message_list.append({"role": "ai", "content": ai_message})
