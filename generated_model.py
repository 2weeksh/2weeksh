import argparse
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from io_utils import load_checkpoint, save_json
from llm_utils import completion_with_retry

# .env파일에서 API 키 불러오기
load_dotenv()

parser = argparse.ArgumentParser(description="뉴스 기사 제목을 생성하는 스크립트")
# 경로 설정
parser.add_argument("--input_path", type=str, default="outputs/rag_retrieval_results.json")
parser.add_argument("--output_dir", type=str, default="outputs/generated", help="모델별 생성 결과를 저장할 폴더")
# 하이퍼파라미터 설정
parser.add_argument("--temperature", type=float, default=1.0, help="제목 생성 temperature")
parser.add_argument("--max_retries", type=int, default=5, help="API 호출 실패 시 최대 재시도 횟수")
parser.add_argument("--save_every", type=int, default=10, help="몇 건마다 중간 저장할지")
args = parser.parse_args()

# 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 온라인 뉴스 콘텐츠를 위한 주목을 끌도록 낚시성 제목을 잘 작성하는 헤드라인 전문가입니다.
주어진 기사 제목과 본문을 바탕으로 사람들의 호기심을 자극하고 클릭을 유도할 수 있는 새로운 제목을 만들어야 합니다.
다만, 제목은 자극적이고 끌리는 방식으로 바꾸되 주제와 관련성을 유지해야 합니다.
생성 형식은 {"title": str} 형식의 JSON 객체 하나만 생성해 주세요.
""".strip()

# LLM프롬프트
NAIVE_LLM_USER_PROMPT = """
### 입력 기사
제목:{query_title}
본문:{query_content}
""".strip()

# RAG 프롬프트
RAG_USER_PROMPT = """
### 입력 기사
제목:{query_title}
본문:{query_content}

### 참고 예시
아래는 위 기사와 관련된 기사들에 대해 사람들의 관심을 끌 수 있도록 재작성한 제목 예시입니다.

{retrieved_title}
""".strip()


def build_messages(model: Dict[str, Any], news: Dict[str, Any]) -> List[Dict[str, str]]:
    # 프롬프트에 들어갈 기사 준비
    query_title = news["query_title"].strip()
    query_content = news["query_content"].strip()

    # rag 사용 여부에 따른 프롬프트 변경
    if model["use_rag"] is True:
        # RAG 참고 기사 정리
        retrieved_title = "\n".join(
            f"- 낚시성 기사 제목: {retrieved['clickbait_title'].strip()}" for retrieved in news["retrieved_articles"]
        )
        user_prompt = RAG_USER_PROMPT.format(
            query_title=query_title, query_content=query_content, retrieved_title=retrieved_title
        )
    else:
        user_prompt = NAIVE_LLM_USER_PROMPT.format(query_title=query_title, query_content=query_content)

    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]


def generate_titles(
    model: Dict[str, Any],
    rag_data: List[Dict[str, Any]],
    output_path: str,
    temperature: float,
    max_retries: int,
    save_every: int,
) -> List[Dict[str, Any]]:
    output = model["output"]

    # 이미 생성된 결과가 있으면 이어서 진행
    results = load_checkpoint(output_path)
    generated_index = {result["index"] for result in results}
    if generated_index:
        print(f"  이전 결과 {len(generated_index)}건을 불러와 이어서 생성합니다.")

    # 기사마다 반복
    for i, news in enumerate(rag_data):
        if i in generated_index:
            continue

        # 모델 호출
        clickbait_title = completion_with_retry(
            model=model["model"],
            messages=build_messages(model, news),
            temperature=temperature,
            max_retries=max_retries,
            response_format={"type": "json_object"},
        )

        # 기본 정보와 생성된 제목 저장
        results.append(
            {
                "index": i,
                "query_title": news["query_title"].strip(),
                "query_content": news["query_content"].strip(),
                "human_direct": news["human_direct_clickbait_title"].strip(),
                output: clickbait_title,
            }
        )

        # 중간 저장 (긴 실행 도중 중단되어도 결과가 남도록)
        if len(results) % save_every == 0:
            save_json(output_path, sorted(results, key=lambda result: result["index"]))
            print(f"  {len(results)}/{len(rag_data)}건 생성 완료")

    return sorted(results, key=lambda result: result["index"])


# 메인 실행
if __name__ == "__main__":

    # 모델 config 값
    models = [
        {"model": "gpt-4o", "use_rag": False, "output": "GPT_LLM"},
        {"model": "gpt-4o", "use_rag": True, "output": "GPT_RAG"},
        {"model": "gemini/gemini-2.0-flash-001", "use_rag": False, "output": "GEMINI_LLM"},
        {"model": "gemini/gemini-2.0-flash-001", "use_rag": True, "output": "GEMINI_RAG"},
    ]

    # 검색 결과 파일 열기
    with open(args.input_path, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # 모델마다 생성 후 각각 저장
    for model in models:
        output_path = os.path.join(args.output_dir, f"{model['output']}.json")
        print(f"[{model['output']}] {model['model']} / RAG={model['use_rag']} 생성 시작")

        results = generate_titles(model, rag_data, output_path, args.temperature, args.max_retries, args.save_every)

        # 생성 결과 저장
        save_json(output_path, results)
        print(f"[{model['output']}] {len(results)}건 저장 -> {output_path}")
