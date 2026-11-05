"""다중 벡터 검색기(Multi-Vector Retriever) 모듈.

요약문은 벡터스토어에 임베딩되어 저장되고, 원본(텍스트·표·이미지)은
docstore에 보관된다. 질의 시 요약으로 검색한 뒤 연결된 원본을 돌려주는
멀티벡터 RAG 패턴을 구현한다.
"""

from __future__ import annotations

import uuid
from typing import List, Sequence

# langchain 1.x 에서 MultiVectorRetriever 는 langchain-classic 으로 분리됨
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings

ID_KEY = "doc_id"


def build_retriever(
    collection_name: str = "multimodal_rag",
    persist_directory: str | None = "chroma_db",
    embedding_model: str = "text-embedding-3-small",
) -> MultiVectorRetriever:
    """빈 다중 벡터 검색기를 생성한다."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=embedding_model),
        persist_directory=persist_directory,
    )
    docstore = InMemoryStore()
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
    )


def _add_documents(
    retriever: MultiVectorRetriever,
    summaries: Sequence[str],
    originals: Sequence[str],
) -> None:
    """요약은 벡터스토어에, 원본은 docstore에 짝지어 추가한다."""
    if not summaries:
        return
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [
        Document(page_content=summary, metadata={ID_KEY: doc_ids[i]})
        for i, summary in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, originals)))


def index_elements(
    retriever: MultiVectorRetriever,
    *,
    text_summaries: Sequence[str],
    texts: Sequence[str],
    table_summaries: Sequence[str],
    tables: Sequence[str],
    image_summaries: Sequence[str],
    images: Sequence[str],
) -> MultiVectorRetriever:
    """세 범주(텍스트·표·이미지)의 요약과 원본을 검색기에 색인한다.

    이미지의 경우 원본으로는 이미지 파일 경로를 저장한다.
    """
    _add_documents(retriever, text_summaries, texts)
    _add_documents(retriever, table_summaries, tables)
    _add_documents(retriever, image_summaries, images)
    return retriever


def query(retriever: MultiVectorRetriever, question: str, k: int = 4) -> List:
    """질문으로 검색해 연결된 원본 요소들을 반환한다."""
    retriever.search_kwargs = {"k": k}
    return retriever.invoke(question)


if __name__ == "__main__":
    r = build_retriever(persist_directory=None)
    index_elements(
        r,
        text_summaries=["매출 15% 증가 요약"],
        texts=["2023년 매출은 전년 대비 15% 증가했다."],
        table_summaries=[],
        tables=[],
        image_summaries=[],
        images=[],
    )
    print(query(r, "매출 성장률은?"))
