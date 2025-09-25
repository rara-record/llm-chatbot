"""Streamlit 앱의 진입점 모듈."""
import streamlit as st

st.set_page_config(page_title="소득세 챗봇", page_icon='🤖')
st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# Initialize chat history
if "message_list" not in st.session_state:
    st.session_state.message_list = [{"role": "assistant", "content": "무엇이 궁금하신가요?"}]

# Display chat messages from history on app rerun
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept user input
if user_question := st.chat_input("소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.chat_message("ai"):
        st.write("여기는 ai 메시지입니다.")
    st.session_state.message_list.append({"role": "ai", "content": "여기는 ai 메시지입니다."})
