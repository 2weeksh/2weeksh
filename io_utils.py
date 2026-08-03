import json
import os
from typing import Any, Dict, List


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
