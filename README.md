# 클릭 유도성 기사 제목 생성 (Clickbait Article Headline Generation)

> 강원대학교 이주혁(지도교수: 강경필)의 학부생 논문 **"대형 언어 모델과 검색 증강 생성 기법을 활용한 클릭 유도성 기사 제목 생성"**의 공식 코드 저장소입니다.
>
> **논문 링크:** [DBpia에서 논문 보기](https://www-dbpia-co-kr.libproxy.kangwon.ac.kr/journal/articleDetail?nodeId=NODE12318738)

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
# 저장소 복제
git clone [https://github.com/2weeksh/clickbait_title_generation_rag.git](https://github.com/2weeksh/clickbait_title_generation_rag.git)
cd clickbait_title_generation_rag

# dev 브랜치로 이동
git checkout dev_clickbait_title_generation_rag
```

### 2. 필요 라이브러리 설치

```
pip install -r requirements.txt
```

### 3. API 키 설정

프로젝트 루트 디렉터리에 `.env` 파일을 생성하고 API 키를 입력하세요.
```
# .env 파일 예시
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="..."
```

## 실험 결과

LLM 자동 평가 및 사용자 설문 평가 결과, Gemini-2.0-Flash (RAG) 모델이 가장 흥미로운 제목을 생성하는 것으로 나타났습니다.

또한 RAG를 적용한 모델들이 Direct 방식보다 더 높은 선호도와 본문 연관성을 보였습니다.
