# 클릭 유도성 기사 제목 생성 (Clickbait Article Headline Generation)

> 강원대학교 이주혁(지도교수: 강경필)의 학부생 논문 **"대형 언어 모델과 검색 증강 생성 기법을 활용한 클릭 유도성 기사 제목 생성"**의 공식 코드 저장소입니다.
>
> 한국정보과학회 2025 한국컴퓨터종합학술대회 논문집, 2025.7, pp. 2036-2038.
> **논문 링크:** [DBpia에서 논문 보기](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12318738)

---

## 프로젝트 개요

본 프로젝트는 디지털 뉴스 환경에서 독자의 흥미를 유발하는 클릭 유도성 기사 제목을 자동으로 생성합니다.

단순 LLM(대형 언어 모델)을 넘어, **RAG(검색 증강 생성)** 기법을 적용하여 기존의 흥미로운 기사 제목을 참고함으로써, 더 자연스럽고 효과적인 제목을 생성하는 것을 목표로 합니다.

## 주요 특징

* **RAG vs Direct 비교:** 기사 본문만 사용하는 'Direct' 방식과, 유사 기사 Top-5를 참조하는 'RAG' 방식을 비교 분석합니다.
* **최신 LLM 활용:** `GPT-4o` 및 `Gemini-2.0-Flash` 모델을 사용하여 제목 생성을 수행합니다.
* **다각적 평가:** LLM 기반 자동 평가(`DeepSeek-V3`)와 사용자 설문 평가를 모두 수행하여 품질을 검증합니다.
* **RAG 효과 입증:** RAG 방식이 Direct 방식보다 **더 높은 선호도**와 **더 낮은 비정합성(본문 연관성)**을 보임을 확인했습니다.

## 사용 기술

* **Language:** `Python`
* **LLM Models:** `GPT-4o`, `Gemini-2.0-Flash`
* **RAG/Search:** `BM25`
* **Evaluation:** `DeepSeek-V3`

## 시스템 아키텍처

본 프로젝트는 Direct 방식과 RAG 방식으로 나누어 제목 생성을 진행합니다.

* **Direct 방식:** LLM(`GPT-4o`, `Gemini-2.0-Flash`)에 기사 본문을 직접 입력하여 제목을 생성합니다.
* **RAG 방식:**
    1.  입력 기사와 유사한 기존 기사 5개를 `BM25` 알고리즘으로 검색합니다.
    2.  (입력 본문 + 검색된 5개 제목)을 프롬프트로 구성하여 LLM에 전달합니다.
    3.  LLM이 참조 정보를 바탕으로 최종 제목을 생성합니다.

## 설치 및 실행 방법

### 1. 저장소 복제 및 브랜치 이동

**중요:** 모든 코드는 `dev_clickbait_title_generation_rag` 브랜치에 있습니다.

```bash
git clone https://github.com/2weeksh/clickbait_title_generation_rag.git
cd clickbait_title_generation_rag
git checkout dev_clickbait_title_generation_rag
```

### 2. 필요 라이브러리 설치

```bash
pip install -r requirements.txt
```

> `litellm` 은 하위 의존성이 Rust 툴체인을 요구해 Windows에서 설치가 실패할 수 있습니다.
> 생성/평가 단계는 Linux 환경에서 실행하는 것을 권장합니다. (`retrieval.py` 는 Windows에서도 동작)

### 3. API 키 설정

`.env.example` 을 `.env` 로 복사한 뒤 키를 채워 넣습니다. (`.env` 는 git에 올라가지 않습니다.)

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=          # 제목 생성 (GPT-4o)
GEMINI_API_KEY=          # 제목 생성 (Gemini-2.0-Flash)
FIREWORKS_AI_API_KEY=    # 평가 (DeepSeek-V3)
```

### 4. 데이터 준비

`data/` 아래에 아래 두 파일이 있어야 합니다. AI Hub 낚시성 기사 탐지 데이터의 **라벨링데이터**를
파일 하나로 병합한 형태이며, 재배포가 제한되어 저장소에는 포함하지 않습니다.

| 파일 | 역할 |
|---|---|
| `TL_Part1_Clickbait_Direct_merged.json` | 검색 코퍼스 — 사람이 작성한 낚시성 제목의 출처 |
| `VL_Part1_Clickbait_Direct_merged.json` | 쿼리 — 제목을 새로 생성할 대상 기사 |

두 파일 모두 아래 키를 가진 객체의 JSON 배열입니다.

| 키 | 설명 |
|---|---|
| `news_title` | 원본(비낚시) 기사 제목 |
| `news_content` | 기사 본문 |
| `new_title` | 사람이 작성한 낚시성 제목 (`Direct_Human` 기준선 및 RAG 참고 예시로 사용) |

> 원천데이터(`TS_*`)는 `sourceDataInfo.newsTitle/newsContent` 만 있고 낚시성 제목이 없으므로
> 코퍼스로 쓸 수 없습니다. 반드시 라벨링데이터를 병합해서 사용하세요.

### 5. 실행

산출물은 `outputs/` 에 저장됩니다. (`data/`, `outputs/` 모두 git 추적 제외)

```bash
# 1) BM25 유사 기사 검색
python retrieval.py --k 5 --sample_size 200

# 2) 모델별 제목 생성 (모델마다 outputs/generated/*.json 로 저장)
python generated_model.py --temperature 1.0

# 3) LLM 심사자 평가
python judge_LLM.py --evaluate_model fireworks_ai/deepseek-v3-0324

# 4) 승률 집계
python aggregate.py
```

2)와 3)은 중간 저장 및 이어하기를 지원합니다. 중단된 경우 같은 명령을 다시 실행하면
이미 처리된 index는 건너뛰고 남은 것부터 이어서 진행합니다.

## 테스트

API 키나 실제 데이터 없이 실행됩니다. 모델 호출은 전부 가짜 함수로 대체됩니다.

```bash
pip install -r requirements-dev.txt
pytest
```

`bm25s` / `kiwipiepy` 가 없는 환경에서는 검색 단계 테스트만 자동으로 건너뜁니다.

| 파일 | 확인 대상 |
|---|---|
| `tests/test_io_utils.py` | 코드펜스·설명이 섞인 LLM 응답의 JSON 추출, 원자적 저장, 손상된 체크포인트 복구 |
| `tests/test_retrieval.py` | 길이 필터, 코퍼스-원본 정렬, `k`/`sample_size` 초과 입력 방어, 시드 재현성 |
| `tests/test_judge.py` | 모델별 결과 병합, 불완전 기사 제외, 후보 제목 파싱, **이어하기 시 제시 순서 재현** |
| `tests/test_aggregate.py` | 제시 위치가 아닌 모델로 승수 귀속, 승률 합 = 1, 파싱 실패 처리, Wilson 신뢰구간 |

## 파이프라인

| 단계 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 검색 | `retrieval.py` | `data/TL_*.json`(코퍼스), `data/VL_*.json`(쿼리) | `outputs/rag_retrieval_results.json` |
| 생성 | `generated_model.py` | 검색 결과 | `outputs/generated/{GPT_LLM,GPT_RAG,GEMINI_LLM,GEMINI_RAG}.json` |
| 평가 | `judge_LLM.py` | 모델별 생성 결과 | `outputs/evaluate_clickbait_results.json` |
| 집계 | `aggregate.py` | 평가 결과 | `outputs/win_rates.json` |

## 평가 방식

`Direct_Human`(사람이 작성한 낚시성 제목), `GPT`, `RAG_GPT`, `Gemini`, `RAG_Gemini` 5개 후보를
매 기사마다 무작위 순서로 제시하고, 심사자가 가장 읽고 싶은 제목 하나를 고릅니다.

* **self-preference 편향 회피:** 심사자는 생성 모델과 겹치지 않는 제3의 모델(`DeepSeek-V3`)을 사용합니다.
* **순서 편향 확인:** `aggregate.py` 가 모델별 승률(Wilson 95% 신뢰구간 포함)과 함께
  **제시 위치별 선택 비율**을 출력하므로, 모든 위치가 0.2 근처인지로 셔플이 제 역할을 했는지 확인할 수 있습니다.

## 실험 결과

LLM 자동 평가 및 사용자 설문 평가 결과, Gemini-2.0-Flash (RAG) 모델이 가장 흥미로운 제목을 생성하는 것으로 나타났습니다.

또한 RAG를 적용한 모델들이 Direct 방식보다 더 높은 선호도와 본문 연관성을 보였습니다.

### DeepSeek-V3 자동 평가 (기사 200건, 후보 5개)

매 기사마다 5개 후보를 무작위 순서로 제시하고 가장 읽고 싶은 제목 하나를 고르게 한 결과입니다.

| 모델 | 선택 비율 | 선택 횟수 |
|---|---:|---:|
| **Gemini-2.0-Flash (RAG)** | **30.5%** | 61 / 200 |
| Gemini-2.0-Flash | 23.5% | 47 / 200 |
| GPT-4o (RAG) | 22.0% | 44 / 200 |
| GPT-4o | 21.5% | 43 / 200 |
| Direct_Human (사람 작성) | 2.5% | 5 / 200 |

사람이 작성한 제목이 2.5%에 그쳐, 심사 모델 기준으로는 생성 제목이 사람 제목보다 강한 클릭 유도력을 보였습니다.

### 설문 기반 인간 평가 (기사 50건, 후보 4개)

자동 평가 대상 200건 중 50건을 무작위 추출해 설문으로 진행했습니다.
설문에서는 사람 작성 제목을 제외한 4개 모델만 비교했습니다.

| 모델 | 선택 비율 |
|---|---:|
| **Gemini-2.0-Flash (RAG)** | **33.4%** |
| Gemini-2.0-Flash | 24.3% |
| GPT-4o (RAG) | 22.3% |
| GPT-4o | 20.0% |

두 평가 모두에서 순위가 같았고, RAG 적용 모델이 각 계열의 기본 모델보다 높은 선택 비율을 기록했습니다.

> 표본이 200건이라 모델 간 차이의 신뢰구간은 겹칩니다. `aggregate.py` 로 계산한 자동 평가의
> 95% 신뢰구간은 Gemini(RAG) [24.5%, 37.2%], GPT-4o [16.4%, 27.7%] 로,
> RAG 계열이 앞선다는 경향은 일관되지만 개별 모델 간 우열을 단정하기에는 표본이 부족합니다.
