"""요약 체인 모듈.

텍스트 / 표 / 이미지 요소를 검색에 적합한 요약으로 변환한다.
요약문은 임베딩되어 벡터스토어에 들어가고, 원본은 docstore에 보관된다
(멀티벡터 검색 패턴). :mod:`retriever` 와 함께 사용한다.
노트북 ``notebooks/Multi_modal_RAG.ipynb`` 의 요약 단계를 모듈화한 것이다.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .extract import encode_image

# ── 텍스트 / 표 요약 ──────────────────────────

_TEXT_PROMPT = """표와 텍스트를 검색용으로 요약하는 도우미 / 임베딩되어 원본 검색에 쓰임 /
검색에 최적화된 간결한 요약 작성 / 요약할 표 또는 텍스트: {element}"""


def generate_text_summaries(
    texts: List[str],
    tables: List[str],
    summarize_texts: bool = False,
    model: str = "gpt-4o-mini",
    max_concurrency: int = 5,
) -> Tuple[List[str], List[str]]:
    """텍스트와 표 요소를 요약한다.

    Args:
        texts: 텍스트 블록(권장: 2k 토큰으로 분할된 것).
        tables: 표 블록.
        summarize_texts: True 면 텍스트도 LLM으로 요약, False 면 원문을 그대로 사용.
        model: 요약에 쓸 채팅 모델.
        max_concurrency: ``batch`` 동시 호출 수 (rate limit 고려).

    Returns:
        (텍스트 요약 리스트, 표 요약 리스트)
    """
    prompt = ChatPromptTemplate.from_template(_TEXT_PROMPT)
    llm = ChatOpenAI(temperature=0, model=model)
    summarize_chain = prompt | llm | StrOutputParser()

    text_summaries: List[str] = []
    table_summaries: List[str] = []
    config = {"max_concurrency": max_concurrency}

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, config=config)
    elif texts:
        text_summaries = texts  # 요약 없이 원문 사용

    if tables:
        table_summaries = summarize_chain.batch(tables, config=config)

    return text_summaries, table_summaries


# ── 이미지 요약 (멀티모달) ────────────────────

_IMAGE_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""


def image_summarize(
    img_base64: str,
    prompt: str = _IMAGE_PROMPT,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
) -> str:
    """base64 이미지 하나를 멀티모달 LLM으로 요약한다."""
    chat = ChatOpenAI(model=model, max_tokens=max_tokens)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
        ])
    ])
    return msg.content


def generate_img_summaries(
    figures_dir: str,
    model: str = "gpt-4o-mini",
) -> Tuple[List[str], List[str]]:
    """figures 폴더의 모든 이미지를 인코딩·요약한다.

    Returns:
        (img_base64_list, image_summaries) — 인코딩 문자열과 요약을 짝으로.
    """
    img_base64_list: List[str] = []
    image_summaries: List[str] = []

    for img_file in sorted(os.listdir(figures_dir)):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(figures_dir, img_file)
            b64 = encode_image(img_path)
            img_base64_list.append(b64)
            image_summaries.append(image_summarize(b64, model=model))

    return img_base64_list, image_summaries


if __name__ == "__main__":
    demo_texts = ["2024년 경제성장률은 전년 대비 둔화되었다."]
    demo_tables = ["| 연도 | 성장률 |\n| 2023 | 1.4 |\n| 2024 | 2.2 |"]
    ts, tabs = generate_text_summaries(demo_texts, demo_tables, summarize_texts=True)
    print("텍스트 요약:", ts)
    print("표 요약:", tabs)
