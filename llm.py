
"""LangChain을 사용한 RAG(Retrieval-Augmented Generation) 체인 구현 모듈."""

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from config import answer_examples


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_llm(model='gpt-4o'):
    """ChatOpenAI 모델 인스턴스를 생성하여 반환한다.

    Parameters:
        model (str): 사용할 OpenAI 모델명. 기본값은 'gpt-4o'.

    Returns:
        ChatOpenAI: 설정된 모델로 초기화된 ChatOpenAI 인스턴스.
    """
    llm = ChatOpenAI(model=model)
    return llm

def get_retriever():
    """Pinecone 벡터 스토어에서 검색을 위한 retriever를 생성하여 반환한다.

    Returns:
        VectorStoreRetriever: Pinecone 벡터 스토어 기반의 검색기 인스턴스.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = 'text-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name, embeddings)

    retriver = database.as_retriever(search_kwargs={'k': 4})
    return retriver

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    # LLM이 대화 맥락을 반영해 새로운 쿼리를 만들 때 사용할 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 대화 맥락을 반영한 retriever 역할을 한다.
    history_aware_retriever = create_history_aware_retriever(
      llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_dictionary_chain():
    """사전 기반 질문 변환을 위한 LangChain 체인을 생성하여 반환한다.

    사용자 질문을 사전에 정의된 규칙에 따라 변환하거나 그대로 반환한다.

    Returns:
        Chain: 사전 기반 질문 변환 체인 인스턴스.
    """
    llm = get_llm()

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
      사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
      만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
      그런 경우 질문만 리턴해주세요
      사전: {dictionary}

      질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain

def get_rag_chain():
    """RAG체인을 생성하여 반환한다.

    문서 검색과 질문 답변을 결합한 체인을 구성한다.

    Returns:
        RetrievalChain: RAG 체인 인스턴스.
    """
    llm = get_llm()


    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples
    )

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = get_history_retriever()

    # 검색된 문서를 프롬프트에 그대로(stuff) 넣어 LLM에 전달하는 체인. (문서량이 많아지면 다른 결합 전략 고려)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Retrieval(검색) → LLM 응답까지 묶는 최상위 체인 빌더.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
      rag_chain,
      get_session_history,
      input_messages_key="input",
      history_messages_key="chat_history",
      output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain

def get_ai_response(user_message):
    """사용자 질문을 받아 RAG 체인을 통해 응답을 생성해 반환한다.

    Parameters:
        user_message (str): 사용자 입력 질문.

    Returns:
        Any: LangChain 체인의 invoke 결과.
    """

    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = { "input": dictionary_chain } | rag_chain
    ai_response = tax_chain.stream(
      { "question": user_message },
      config={
        "configurable": {"session_id": "abc123"}
      },
    )

    return ai_response
