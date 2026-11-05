"""PDF 파싱 모듈.

`unstructured` 를 사용해 PDF를 텍스트 / 표 / 이미지 요소로 분해한다.
이미지는 ``data/figures`` 아래에 저장되고, 함수는 요소를 범주별로 분류해
(texts, tables, images) 형태로 돌려준다.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from unstructured.partition.pdf import partition_pdf


@dataclass
class ExtractedElements:
    """PDF에서 추출한 요소를 범주별로 담는 컨테이너."""

    texts: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)  # 이미지 파일 경로 목록

    def summary(self) -> str:
        return (
            f"texts={len(self.texts)}, "
            f"tables={len(self.tables)}, "
            f"images={len(self.images)}"
        )


def extract_pdf_elements(
    pdf_path: str | Path,
    figures_dir: str | Path = "data/figures",
) -> ExtractedElements:
    """PDF를 파싱해 텍스트·표·이미지 요소로 분해한다.

    Args:
        pdf_path: 원본 PDF 경로.
        figures_dir: 추출한 이미지를 저장할 디렉터리.

    Returns:
        범주별로 분류된 :class:`ExtractedElements`.
    """
    pdf_path = Path(pdf_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {pdf_path}")

    # 고해상도 전략으로 표·이미지까지 함께 추출한다.
    raw_elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        extract_image_block_output_dir=str(figures_dir),
        # 텍스트를 의미 단위로 묶기 위한 청킹 옵션
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

    result = ExtractedElements()
    for element in raw_elements:
        category = str(type(element))
        if "Table" in category:
            result.tables.append(str(element))
        elif "Image" in category:
            # 이미지 요소는 파일로 저장되므로 경로를 수집한다.
            metadata = getattr(element, "metadata", None)
            image_path = getattr(metadata, "image_path", None)
            if image_path:
                result.images.append(image_path)
        else:
            result.texts.append(str(element))

    # extract_image_block_output_dir 로 저장된 이미지도 스캔해 보강한다.
    for img in sorted(figures_dir.glob("*")):
        if img.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if str(img) not in result.images:
                result.images.append(str(img))

    return result


def encode_image_to_base64(image_path: str | Path) -> str:
    """이미지 파일을 멀티모달 LLM 입력용 base64 문자열로 인코딩한다."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


if __name__ == "__main__":
    # 간단한 수동 실행 예시
    pdf = os.environ.get("PDF_PATH", "data/invest.pdf")
    elements = extract_pdf_elements(pdf)
    print("추출 결과:", elements.summary())
