"""요약 체인 모듈.

텍스트 / 표 / 이미지 요소를 검색에 적합한 짧은 요약으로 변환한다.
요약문은 임베딩되어 벡터스토어에 들어가고, 원본은 docstore에 보관된다
(멀티벡터 검색 패턴). :mod:`retriever` 와 함께 사용한다.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .extract import encode_image_to_base64

# ── 텍스트 / 표 요약 ──────────────────────────

_TEXT_PROMPT = """당신은 검색용 요약을 작성하는 도우미입니다.
아래 내용을 검색에 최적화된 간결한 요약으로 만들어 주세요.
표라면 핵심 수치와 항목을 빠짐없이 담되 설명은 짧게 유지하세요.

내용:
{element}"""


def summarize_texts(
    texts: List[str],
    tables: List[str],
    model: str = "gpt-4o-mini",
    max_concurrency: int = 5,
) -> tuple[List[str], List[str]]:
    """텍스트와 표 요소를 각각 요약한다.

    Returns:
        (텍스트 요약 리스트, 표 요약 리스트)
    """
    prompt = ChatPromptTemplate.from_template(_TEXT_PROMPT)
    llm = ChatOpenAI(model=model, temperature=0)
    chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    config = {"max_concurrency": max_concurrency}
    text_summaries = chain.batch(texts, config=config) if texts else []
    table_summaries = chain.batch(tables, config=config) if tables else []
    return text_summaries, table_summaries


# ── 이미지 요약 (멀티모달) ────────────────────

_IMAGE_PROMPT = (
    "이 이미지를 검색에 활용할 수 있도록 상세히 설명하세요. "
    "차트·그래프라면 축, 추세, 핵심 수치를 포함하세요."
)


def summarize_images(
    image_paths: List[str],
    model: str = "gpt-4o-mini",
) -> List[str]:
    """이미지 요소를 멀티모달 LLM으로 요약한다."""
    if not image_paths:
        return []

    llm = ChatOpenAI(model=model, temperature=0, max_tokens=1024)
    summaries: List[str] = []

    for path in image_paths:
        b64 = encode_image_to_base64(path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": _IMAGE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ]
        )
        summaries.append(llm.invoke([message]).content)

    return summaries


if __name__ == "__main__":
    demo_texts = ["2023년 매출은 전년 대비 15% 증가했다."]
    demo_tables = ["| 분기 | 매출 |\n| Q1 | 100 |\n| Q2 | 115 |"]
    ts, tabs = summarize_texts(demo_texts, demo_tables)
    print("텍스트 요약:", ts)
    print("표 요약:", tabs)
