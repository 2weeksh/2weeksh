import argparse
import json
import math
from collections import Counter
from typing import Any, Dict, List

from io_utils import parse_json_object, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="평가 결과를 모델별 승률로 집계하는 스크립트")
    # 경로 설정
    parser.add_argument("--input_path", type=str, default="outputs/evaluate_clickbait_results.json")
    parser.add_argument("--output_path", type=str, default="outputs/win_rates.json")
    return parser


# 평가에 사용된 라벨
LABELS = ["A", "B", "C", "D", "E"]


def parse_choice(evaluation: Any, valid_labels: List[str]) -> str:
    """평가 모델 출력에서 선택한 라벨만 꺼낸다. 형식이 어긋나면 빈 문자열을 반환한다."""
    parsed = parse_json_object(evaluation)
    if parsed is None:
        return ""

    choice = parsed.get("choice")
    if not isinstance(choice, str):
        return ""

    choice = choice.strip().upper()
    return choice if choice in valid_labels else ""


def wilson_interval(wins: int, total: int, z: float = 1.96) -> Dict[str, float]:
    """승률의 95% 신뢰구간(Wilson score interval). 표본이 작을 때도 정규근사보다 안정적이다."""
    if total == 0:
        return {"lower": 0.0, "upper": 0.0}

    phat = wins / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator

    return {"lower": round(max(0.0, center - margin), 4), "upper": round(min(1.0, center + margin), 4)}


def aggregate(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    win_counts: Counter = Counter()  # 모델별 선택된 횟수
    position_counts: Counter = Counter()  # 라벨(제시 위치)별 선택된 횟수 - 순서 편향 확인용
    appearance_counts: Counter = Counter()  # 모델별 등장 횟수 (모든 기사에 5개가 다 나오면 전부 동일)
    invalid_count = 0

    for item in evaluation_results:
        shuffled_order = item["shuffled_order"]
        for model_name in shuffled_order.values():
            appearance_counts[model_name] += 1

        choice = parse_choice(item["evaluation"], list(shuffled_order.keys()))
        if not choice:
            invalid_count += 1
            continue

        win_counts[shuffled_order[choice]] += 1
        position_counts[choice] += 1

    valid_total = sum(win_counts.values())

    # 모델별 승률
    model_stats = []
    for model_name, appearances in appearance_counts.items():
        wins = win_counts[model_name]
        model_stats.append(
            {
                "model": model_name,
                "wins": wins,
                "appearances": appearances,
                "win_rate": round(wins / valid_total, 4) if valid_total else 0.0,
                "ci95": wilson_interval(wins, valid_total),
            }
        )
    model_stats.sort(key=lambda stat: stat["wins"], reverse=True)

    # 순서 편향 확인 (모든 위치가 1/5 = 0.2에 가까워야 셔플이 제 역할을 한 것)
    position_stats = {
        label: {
            "wins": position_counts[label],
            "rate": round(position_counts[label] / valid_total, 4) if valid_total else 0.0,
        }
        for label in LABELS
    }

    return {
        "total_items": len(evaluation_results),
        "valid_items": valid_total,
        "invalid_items": invalid_count,
        "model_stats": model_stats,
        "position_stats": position_stats,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"\n전체 {summary['total_items']}건 / 유효 {summary['valid_items']}건 / 파싱 실패 {summary['invalid_items']}건")

    print("\n[모델별 승률]")
    print(f"{'모델':<14}{'승':>6}{'승률':>10}{'95% CI':>20}")
    for stat in summary["model_stats"]:
        ci = stat["ci95"]
        ci_text = f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
        print(f"{stat['model']:<14}{stat['wins']:>6}{stat['win_rate']:>10.3f}{ci_text:>20}")

    print("\n[제시 위치별 선택 비율] - 편향이 없다면 모두 0.200 근처")
    for label, stat in summary["position_stats"].items():
        print(f"  {label}: {stat['rate']:.3f} ({stat['wins']}회)")


# 메인 실행
def main() -> None:
    args = build_parser().parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        evaluation_results = json.load(f)

    summary = aggregate(evaluation_results)
    print_summary(summary)

    # 집계 결과 저장
    save_json(args.output_path, summary)
    print(f"\n집계 결과 저장 -> {args.output_path}")


if __name__ == "__main__":
    main()
