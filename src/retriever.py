"""다중 벡터 검색기(Multi-Vector Retriever) + 멀티모달 RAG 체인 모듈.

요약문은 벡터스토어에 임베딩되어 저장되고, 원본(텍스트·표·이미지 base64)은
docstore에 보관된다. 질의 시 요약으로 검색한 뒤 연결된 원본을 꺼내
멀티모달 LLM에 넣어 답변을 만든다.
노트북 ``notebooks/Multi_modal_RAG.ipynb`` 의 색인·검색·체인 단계를 모듈화한 것이다.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List

# langchain 1.x 에서 MultiVectorRetriever 는 langchain-classic 으로 분리됨
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ID_KEY = "doc_id"
# base64 이미지 판별용 정규식: 영문/숫자/+// 가 100자 이상 이어지고 끝에 = 0~2개
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")


def build_retriever(
    collection_name: str = "multimodal_rag",
    embedding_model: str = "text-embedding-3-small",
    persist_directory: str | None = None,
) -> MultiVectorRetriever:
    """요약=벡터스토어, 원본=docstore 로 구성된 다중 벡터 검색기를 만든다."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=embedding_model),
        persist_directory=persist_directory,
    )
    store = InMemoryStore()
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=ID_KEY,
    )


def add_documents(
    retriever: MultiVectorRetriever,
    doc_summaries: List[str],
    doc_contents: List,
) -> None:
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장한다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def index_elements(
    retriever: MultiVectorRetriever,
    *,
    text_summaries: List[str],
    texts: List[str],
    table_summaries: List[str],
    tables: List[str],
    image_summaries: List[str],
    img_base64_list: List[str],
) -> MultiVectorRetriever:
    """세 범주(텍스트·표·이미지)의 요약과 원본을 검색기에 색인한다.

    주의: 같은 데이터로 두 번 호출하면 중복 색인된다.
    """
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, img_base64_list)
    return retriever


# ── 검색 결과 분리 & 멀티모달 프롬프트 ─────────

def split_image_text_types(docs) -> Dict[str, List[str]]:
    """검색 결과를 base64 이미지와 텍스트로 나눈다."""
    b64_images: List[str] = []
    texts_out: List[str] = []

    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        if _BASE64_RE.match(content):
            b64_images.append(content)
        else:
            texts_out.append(content)

    return {"images": b64_images, "texts": texts_out}


def img_prompt_func(data_dict: Dict) -> List[HumanMessage]:
    """검색된 원본을 멀티모달 LLM이 읽을 메시지로 조립한다.

    이미지가 없어도 항상 메시지를 반환한다 (None 반환 방지).
    """
    messages: List[Dict] = []

    # 1) 이미지를 먼저 넣는다 (모델이 그림을 먼저 보게)
    for image in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })

    # 2) 텍스트 컨텍스트와 질문을 넣는다
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
        "type": "text",
        "text": ("아래 자료를 근거로 질문에 한국어로 답하세요.\n"
                 "자료에 없는 내용은 모른다고 답하세요.\n\n"
                 f"질문: {data_dict['question']}\n\n"
                 f"참고 텍스트:\n{formatted_texts}"),
    })

    return [HumanMessage(content=messages)]


def build_multimodal_chain(
    retriever: MultiVectorRetriever,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
):
    """검색 → 이미지/텍스트 분리 → 멀티모달 프롬프트 → LLM → 문자열 체인."""
    llm = ChatOpenAI(temperature=0, model=model, max_tokens=max_tokens)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    r = build_retriever()
    index_elements(
        r,
        text_summaries=["2024년 경제성장률 요약"],
        texts=["2024년 경제성장률은 2.2%로 전망된다."],
        table_summaries=[],
        tables=[],
        image_summaries=[],
        img_base64_list=[],
    )
    chain = build_multimodal_chain(r)
    print(chain.invoke("2024년 경제성장률은?"))
