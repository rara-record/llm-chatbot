
"""LangChain을 사용한 RAG(Retrieval-Augmented Generation) 체인 구현 모듈."""

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


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
    retriever = get_retriever()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

def get_ai_message(user_message):
    """사용자 질문을 받아 RAG 체인을 통해 응답을 생성해 반환한다.

    Parameters:
        user_message (str): 사용자 입력 질문.

    Returns:
        Any: LangChain 체인의 invoke 결과.
    """

    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = { "input": dictionary_chain } | rag_chain
    response_message = tax_chain.invoke({ "question": user_message })

    return response_message["answer"]
