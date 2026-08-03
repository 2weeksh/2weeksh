# RAG 기반 낚시성 기사 제목 생성 및 평가

BM25로 검색한 사람이 작성한 낚시성 제목을 참고 예시로 넣어(RAG), LLM이 생성한 기사 제목이
얼마나 클릭을 유도하는지를 LLM 심사자로 평가하는 실험 코드입니다.

## 논문

> 이주혁, 강경필. **대형 언어 모델과 검색 증강 생성 기법을 활용한 클릭 유도성 기사 제목 생성**.
> 한국정보과학회 2025 한국컴퓨터종합학술대회 논문집, 2025.7, pp. 2036-2038.
> <https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12318738>

## 설치

```bash
pip install -r requirements.txt
cp .env.example .env   # .env 에 API 키 입력
```

> `litellm` 은 하위 의존성이 Rust 툴체인을 요구해 Windows에서 설치가 실패할 수 있습니다.
> 생성/평가 단계는 Linux 환경에서 실행하는 것을 권장합니다. (`retrival.py` 는 Windows에서도 동작)

## 데이터

`data/` 아래에 아래 두 파일이 있어야 합니다. (AI Hub 낚시성 기사 탐지 데이터의 **라벨링데이터**를
파일 하나로 병합한 형태이며, 용량이 커서 git에는 포함하지 않습니다.)

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

## 실행

데이터는 `data/` 에, 산출물은 `outputs/` 에 놓는 것을 기본값으로 합니다. (둘 다 git 추적 제외)

```bash
# 1) BM25 유사 기사 검색
python retrival.py --k 5 --sample_size 200

# 2) 모델별 제목 생성 (모델마다 outputs/generated/*.json 로 저장)
python generated_model.py --temperature 1.0

# 3) LLM 심사자 평가
python judge_LLM.py --evaluate_model fireworks_ai/deepseek-v3-0324

# 4) 승률 집계
python aggregate.py
```

2)와 3)은 중간 저장 및 이어하기를 지원합니다. 중단된 경우 같은 명령을 다시 실행하면
이미 처리된 index는 건너뛰고 남은 것부터 이어서 진행합니다.

## 파이프라인

| 단계 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 검색 | `retrival.py` | `data/TL_*.json`(코퍼스), `data/VL_*.json`(쿼리) | `outputs/rag_retrieval_results.json` |
| 생성 | `generated_model.py` | 검색 결과 | `outputs/generated/{GPT_LLM,GPT_RAG,GEMINI_LLM,GEMINI_RAG}.json` |
| 평가 | `judge_LLM.py` | 모델별 생성 결과 | `outputs/evaluate_clickbait_results.json` |
| 집계 | `aggregate.py` | 평가 결과 | `outputs/win_rates.json` |

## 비교 대상

`Direct_Human`(사람이 작성한 낚시성 제목), `GPT`, `RAG_GPT`, `Gemini`, `RAG_Gemini` 5개 후보를
매 기사마다 무작위 순서로 제시하고, 심사자가 가장 읽고 싶은 제목 하나를 고릅니다.
심사자는 생성 모델과 겹치지 않는 제3의 모델(DeepSeek-V3)을 사용해 self-preference 편향을 피했습니다.
`aggregate.py` 는 모델별 승률과 함께 **제시 위치별 선택 비율**을 출력하므로 순서 편향 여부를 확인할 수 있습니다.
