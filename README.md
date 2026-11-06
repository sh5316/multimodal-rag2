# Multimodal RAG

PDF 문서의 **텍스트·표·이미지**를 함께 검색·활용하는 멀티모달 RAG 파이프라인.
경제 보고서(`data/economy2024.pdf`)를 예시 데이터로 사용한다.

---

## 1. 문제 정의

일반적인 RAG는 PDF에서 **텍스트만** 추출해 색인한다. 하지만 보고서·논문 같은
문서는 핵심 정보가 **표와 그래프(이미지)** 에 담겨 있는 경우가 많다.
텍스트만 쓰면 아래와 같은 질문에 제대로 답하지 못한다.

- "그래프에서 2024년 성장률 추세가 어떻게 나타나나요?"
- "표에 정리된 연도별 수치를 요약해 주세요."

**목표:** PDF를 텍스트·표·이미지 세 범주로 분해하고, 각 요소를 요약해
하나의 벡터 공간에서 검색한 뒤, 원본(표·이미지 포함)을 멀티모달 LLM에
넣어 답변하는 파이프라인을 만든다.

---

## 2. 아키텍처


**모듈 구성** (`src/`)

| 모듈 | 역할 | 핵심 함수 |
|------|------|-----------|
| `extract.py` | PDF를 텍스트·표·이미지로 분해, 토큰 단위 재분할, 이미지 base64 인코딩 | `extract_pdf_elements`, `categorize_elements`, `split_texts_to_tokens` |
| `summarize.py` | 각 요소를 검색용 요약으로 변환 (이미지는 멀티모달 LLM으로 요약) | `generate_text_summaries`, `generate_img_summaries` |
| `retriever.py` | 멀티벡터 검색기 구성, 색인, 멀티모달 RAG 체인 조립 | `build_retriever`, `index_elements`, `build_multimodal_chain` |

**사용 기술**

- 파싱: `unstructured` (`hi_res` 전략, 표 구조 추론 + 이미지 추출)
- LLM / 임베딩: OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- 벡터스토어: Chroma / 오케스트레이션: LangChain 1.x (+ `langchain-classic`)

전체 흐름의 원본 노트북은 `notebooks/Multi_modal_RAG.ipynb` 이며,
`src/` 는 이를 재사용 가능하게 모듈화한 것이다.

---

## 3. 실행법

### 설치

```bash
git clone <저장소 주소>
cd multimodal-rag2

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt     # 또는 requirements.lock.txt

# 커널 등록 (노트북 사용 시)
python -m ipykernel install --user --name multimodal-rag

# .env 작성
cp .env.example .env                # 윈도우: copy .env.example .env
# .env 파일을 열어 실제 OPENAI_API_KEY 입력
```

> `unstructured[all-docs]` 의 `hi_res` 전략은 시스템에 **Poppler** 와
> **Tesseract** 가 설치되어 있어야 한다. (OS별 설치 방법은 해당 패키지 문서 참고)

### 노트북으로 실행

```bash
jupyter lab
# notebooks/Multi_modal_RAG.ipynb 를 열어 위에서부터 순서대로 실행
```

### 모듈로 실행 (파이프라인 예시)

```python
from src.extract import (
    extract_pdf_elements, categorize_elements, split_texts_to_tokens,
)
from src.summarize import generate_text_summaries, generate_img_summaries
from src.retriever import build_retriever, index_elements, build_multimodal_chain

FIGURES = "data/figures"

# 1) 추출
raw = extract_pdf_elements("data/economy2024.pdf", FIGURES)
texts, tables = categorize_elements(raw)
chunks = split_texts_to_tokens(texts)

# 2) 요약
text_summaries, table_summaries = generate_text_summaries(
    chunks, tables, summarize_texts=True,
)
img_base64_list, image_summaries = generate_img_summaries(FIGURES)

# 3) 색인
retriever = build_retriever()
index_elements(
    retriever,
    text_summaries=text_summaries, texts=chunks,
    table_summaries=table_summaries, tables=tables,
    image_summaries=image_summaries, img_base64_list=img_base64_list,
)

# 4) 질의
chain = build_multimodal_chain(retriever)
print(chain.invoke("2024년 경제성장률 전망은?"))
```

각 모듈은 단독 실행용 데모도 포함한다 (`python -m src.extract` 등).

---

## 5. 예시 질문 (Q&A)

파이프라인에 실제로 던진 질문과 답변을 정리한다.
(텍스트·표·이미지 중 어느 요소가 근거로 쓰였는지 함께 기록하면 좋다.)

### Q1.

> **질문:**
>
> **답변:**
>
> **근거:** <!-- 텍스트 / 표 / 이미지(figure-xx) -->

### Q2.

> **질문:**
>
> **답변:**
>
> **근거:**

### Q3.

> **질문:**
>
> **답변:**
>
> **근거:**

### Q4.

> **질문:**
>
> **답변:**
>
> **근거:**

### Q5.

> **질문:**
>
> **답변:**
>
> **근거:**

---

## 6. 한계

- **원본 저장소가 휘발성이다.** `retriever.py` 는 원본을 `InMemoryStore`
  에 보관하므로 **프로세스가 끝나면 사라진다.** 벡터스토어(Chroma)만
  `persist_directory` 로 영속화되어, 재시작 시 요약 검색은 되지만 원본을
  꺼낼 수 없다. 운영하려면 영속 docstore(예: 파일/DB 기반)가 필요하다.

- **중복 색인 방지가 없다.** `index_elements` 를 같은 데이터로 두 번
  호출하면 그대로 중복 색인된다.

- **이미지 판별이 휴리스틱이다.** 검색 결과에서 이미지/텍스트를
  정규식(base64 패턴)으로 구분하므로, 특수한 입력에서 오분류 가능성이 있다.

- **외부 API/시스템 의존성이 크다.** OpenAI API 키가 필요하고(비용 발생),
  `hi_res` 파싱은 Poppler·Tesseract 설치에 의존한다. 파싱 품질은
  PDF 레이아웃에 따라 편차가 크다.

- **단일 문서 기준으로 설계되었다.** 예시는 경제 보고서 1건이며,
  대규모 문서 집합·메타데이터 필터링·출처 표기 등은 포함하지 않는다.

- **평가 체계가 없다.** 검색 정확도·답변 품질을 자동으로 측정하는
  평가셋이나 지표가 아직 없다.
