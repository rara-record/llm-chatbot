"""Streamlit 앱의 진입점 모듈."""
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

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

def get_ai_message(user_message):
    """사용자 질문을 받아 RAG 체인을 통해 응답을 생성해 반환한다.

    Parameters:
        user_message (str): 사용자 입력 질문.

    Returns:
        Any: LangChain 체인의 invoke 결과.
    """

    load_dotenv()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = 'text-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name, embeddings)

    llm = ChatOpenAI(model='gpt-4o')
    prompt = hub.pull("rlm/rag-prompt")
    retriver = database.as_retriever(search_kwargs={'k': 4})

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
      사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
      만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
      그런 경우 질문만 리턴해주세요
      사전: {dictionary}

      질문: {{question}}
    """)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(retriver, combine_docs_chain)

    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = { "input": dictionary_chain } | qa_chain
    response_message = tax_chain.invoke({ "question": user_message })

    return response_message["answer"]


# Accept user input
if user_question := st.chat_input("소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
