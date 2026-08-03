import json
import os
import re
from typing import Any, Dict, List, Optional

# 모델이 JSON을 코드펜스로 감싸거나 앞뒤에 설명을 붙이는 경우를 위한 패턴
FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    """LLM 응답에서 JSON 객체를 꺼낸다.

    ```json ... ``` 로 감싸거나 앞뒤에 설명 문장이 붙어 오는 경우가 잦아서,
    원문 -> 코드펜스 내부 -> 중괄호 구간 순으로 파싱을 시도한다.
    끝내 실패하면 None 을 반환한다.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None

    candidates = [raw]
    fenced = FENCED_JSON_PATTERN.search(raw)
    if fenced:
        candidates.append(fenced.group(1))
    braced = JSON_OBJECT_PATTERN.search(raw)
    if braced:
        candidates.append(braced.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def load_checkpoint(path: str) -> List[Dict[str, Any]]:
    """중간 저장 파일이 있으면 불러오고, 없으면 빈 리스트를 반환한다."""
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"  [경고] 중간 저장 파일이 손상되어 무시합니다: {path}")
        return []


def save_json(path: str, data: Any) -> None:
    """결과를 임시 파일에 먼저 쓴 뒤 교체해서, 저장 중 중단되어도 기존 파일이 깨지지 않게 한다."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)
