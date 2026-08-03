import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from io_utils import load_checkpoint, parse_json_object, save_json
from llm_utils import completion_with_retry

parser = argparse.ArgumentParser(description="생성된 제목들을 평가하는 스크립트")
# 경로 설정
parser.add_argument("--input_dir", type=str, default="outputs/generated", help="모델별 생성 결과가 저장된 폴더")
parser.add_argument("--output_path", type=str, default="outputs/evaluate_clickbait_results.json")
# 하이퍼파라미터 설정
parser.add_argument("--evaluate_model", type=str, default="fireworks_ai/deepseek-v3-0324")
parser.add_argument("--temperature", type=float, default=0.0, help="평가 모델 temperature")
parser.add_argument("--max_retries", type=int, default=5, help="API 호출 실패 시 최대 재시도 횟수")
parser.add_argument("--save_every", type=int, default=10, help="몇 건마다 중간 저장할지")
args = parser.parse_args()

# API 키 설정
load_dotenv()

# 평가 모델 시스템 프롬프트
EVALUATOR_SYSTEM_PROMPT = """
당신은 흥미로운 기사 제목을 선택해서 읽는 신문 구독자입니다.

아래에 제시된 제목들에 대해 가장 흥미롭고 읽고 싶은 제목을 선택해 주세요.
여기서 제목은 무작위 순서로 제시되며, 순서 편향 없이 선택해 주세요.

반드시 아래 형식처럼 JSON으로 출력해 주세요:
{"choice": "선택한 알파벳"}
""".strip()

# 평가 모델 유저 프롬프트
EVALUATOR_USER_PROMPT = """
### 제목 후보
A: {A}
B: {B}
C: {C}
D: {D}
E: {E}
""".strip()

# 원본 제목 키와 평가용 라벨 매핑
KEY_TO_LABEL_MAP = {"human_direct": "A", "GPT_LLM": "B", "GPT_RAG": "C", "GEMINI_LLM": "D", "GEMINI_RAG": "E"}

# 평가 라벨과 모델 이름 매핑
LABEL_TO_MODEL_MAP = {"A": "Direct_Human", "B": "GPT", "C": "RAG_GPT", "D": "Gemini", "E": "RAG_Gemini"}

# 셔플 후 사용할 라벨
LABELS = ["A", "B", "C", "D", "E"]


def load_generated(input_dir: str) -> List[Dict[str, Any]]:
    """모델별 생성 파일을 index 기준으로 합친다. 모든 모델의 제목이 갖춰진 기사만 평가 대상으로 삼는다."""
    model_keys = [key for key in KEY_TO_LABEL_MAP if key != "human_direct"]
    merged: Dict[int, Dict[str, Any]] = {}

    for key in model_keys:
        path = os.path.join(input_dir, f"{key}.json")
        if not os.path.exists(path):
            raise SystemExit(
                f"[오류] 생성 결과 파일이 없습니다: {path}\n"
                f"       generated_model.py 를 먼저 실행해 {len(model_keys)}개 모델의 제목을 모두 생성하세요."
            )
        with open(path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                row = merged.setdefault(item["index"], {"index": item["index"]})
                row.setdefault("query_title", item["query_title"])
                row.setdefault("human_direct", item["human_direct"])
                row[key] = item[key]

    # 일부 모델만 생성된 기사는 공정한 비교가 안 되므로 제외
    complete = [merged[index] for index in sorted(merged) if all(key in merged[index] for key in model_keys)]
    if len(complete) < len(merged):
        print(f"[경고] 모델별 생성 개수가 달라 {len(merged) - len(complete)}건을 평가에서 제외했습니다.")

    return complete


def parse_title(raw_title_json: Any) -> Tuple[str, bool]:
    """생성 결과 JSON에서 제목만 꺼낸다. 파싱에 실패하면 원문을 그대로 쓰고 실패를 알린다."""
    parsed = parse_json_object(raw_title_json)
    if parsed is not None:
        title = parsed.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip(), True

    return str(raw_title_json).strip(), False


# 평가 함수
def evaluation(
    all_data: List[Dict[str, Any]],
    evaluate_model: str,
    output_path: str,
    temperature: float,
    max_retries: int,
    save_every: int,
) -> List[Dict[str, Any]]:
    # 이미 평가된 결과가 있으면 이어서 진행
    evaluation_results = load_checkpoint(output_path)
    evaluated_index = {result["index"] for result in evaluation_results}
    if evaluated_index:
        print(f"이전 결과 {len(evaluated_index)}건을 불러와 이어서 평가합니다.")

    parse_failure = 0

    # 기사마다 평가
    for item in all_data:
        # 제목들 라벨과 같이 정리
        candidates_with_labels = []
        item_parse_failure = 0

        # 셔플을 위한 반복문
        for key, label in KEY_TO_LABEL_MAP.items():
            # human_direct는 사람이 작성한 평문 제목이라 JSON 파싱 대상이 아니다
            if key == "human_direct":
                candidates_with_labels.append((label, str(item.get(key, "내용 없음")).strip()))
                continue

            title, parsed = parse_title(item.get(key, '{"title": "내용 없음"}'))
            if not parsed:
                item_parse_failure += 1
            candidates_with_labels.append((label, title))

        # 평가 순서 섞기 (건너뛴 기사도 셔플을 소모해야 재실행 시 순서가 동일하게 유지됨)
        random.shuffle(candidates_with_labels)

        if item["index"] in evaluated_index:
            continue

        # 실제로 평가한 기사만 집계해야 이어하기 때 경고 건수가 부풀지 않는다
        parse_failure += item_parse_failure

        # 섞인 순서에 따라 데이터 재구성
        shuffled_candidates_for_prompt = {
            new_label: title for new_label, (_, title) in zip(LABELS, candidates_with_labels)
        }
        shuffled_order = {
            new_label: LABEL_TO_MODEL_MAP[old_label]
            for new_label, (old_label, _) in zip(LABELS, candidates_with_labels)
        }

        # 랜덤 적용 프롬프트 생성
        user_prompt = EVALUATOR_USER_PROMPT.format(**shuffled_candidates_for_prompt)
        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 평가 모델 호출
        result_text = completion_with_retry(
            model=evaluate_model,
            messages=messages,
            temperature=temperature,
            max_retries=max_retries,
            response_format={"type": "json_object"},
        )

        # 결과 저장 양식
        evaluation_results.append(
            {
                "index": item["index"],
                "evaluation": result_text,
                "shuffled_order": shuffled_order,
                "shuffled_titles": shuffled_candidates_for_prompt,
            }
        )

        # 중간 저장
        if len(evaluation_results) % save_every == 0:
            save_json(output_path, sorted(evaluation_results, key=lambda result: result["index"]))
            print(f"{len(evaluation_results)}/{len(all_data)}건 평가 완료")

    if parse_failure:
        print(f"[경고] 생성 제목 JSON 파싱 실패 {parse_failure}건 (원문을 그대로 후보에 사용했습니다)")

    return sorted(evaluation_results, key=lambda result: result["index"])


# 메인 실행
if __name__ == "__main__":
    random.seed(42)

    all_data = load_generated(args.input_dir)
    print(f"평가 대상 {len(all_data)}건 / 평가 모델 {args.evaluate_model}")

    evaluation_results = evaluation(
        all_data, args.evaluate_model, args.output_path, args.temperature, args.max_retries, args.save_every
    )

    # 최종 결과 저장
    save_json(args.output_path, evaluation_results)
    print(f"{len(evaluation_results)}건 저장 -> {args.output_path}")
