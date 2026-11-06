"""PDF 파싱 모듈.

`unstructured` 로 PDF를 텍스트 / 표 / 이미지 요소로 분해하고,
텍스트는 검색에 알맞은 크기(약 2k 토큰)로 재분할한다.
노트북 ``notebooks/Multi_modal_RAG.ipynb`` 의 추출 단계를 모듈화한 것이다.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(pdf_path: str | Path, figures_dir: str | Path):
    """PDF에서 이미지·표·텍스트 블록을 추출한다.

    Args:
        pdf_path: 원본 PDF 파일 경로.
        figures_dir: 추출한 이미지를 저장할 폴더. 없으면 생성한다.

    Returns:
        ``unstructured`` 요소 리스트 (raw_pdf_elements).
    """
    pdf_path = Path(pdf_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {pdf_path}")

    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                     # 표·이미지까지 추출
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=str(figures_dir),
    )


def categorize_elements(raw_pdf_elements) -> Tuple[List[str], List[str]]:
    """추출 요소를 (텍스트, 표) 로 분류한다.

    - ``Table``            → 표
    - ``CompositeElement`` → 텍스트
    """
    texts: List[str] = []
    tables: List[str] = []

    for element in raw_pdf_elements:
        type_name = str(type(element))
        if "unstructured.documents.elements.Table" in type_name:
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in type_name:
            texts.append(str(element))

    return texts, tables


def split_texts_to_tokens(
    texts: List[str],
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[str]:
    """텍스트 블록들을 토큰 기준으로 재분할한다 (기본 약 2k 토큰).

    tiktoken 인코더를 사용하므로 임베딩·LLM 입력 길이에 맞추기 좋다.
    """
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    joined = " ".join(texts)
    return splitter.split_text(joined)


def list_image_files(figures_dir: str | Path) -> List[Path]:
    """figures 폴더에서 이미지 파일 경로를 이름순으로 반환한다."""
    figures_dir = Path(figures_dir)
    if not figures_dir.exists():
        return []
    return sorted(
        f for f in figures_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def encode_image(image_path: str | Path) -> str:
    """이미지 파일을 멀티모달 LLM 입력용 base64 문자열로 인코딩한다."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


if __name__ == "__main__":
    # 수동 실행 예시: PDF → 분류 → 토큰 분할
    root = Path(__file__).resolve().parent.parent
    pdf = os.environ.get("PDF_PATH", str(root / "data" / "economy2024.pdf"))
    figures = root / "data" / "figures"

    raw = extract_pdf_elements(pdf, figures)
    texts, tables = categorize_elements(raw)
    chunks = split_texts_to_tokens(texts)
    print(f"texts={len(texts)}, tables={len(tables)}, chunks={len(chunks)}, "
          f"images={len(list_image_files(figures))}")
