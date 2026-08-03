"""테스트 공통 설정.

API 키나 실제 데이터 없이 돌아가도록, 모델 호출은 전부 가짜 함수로 대체한다.
"""
import json
import os
import sys
from typing import Any, Dict, List

import pytest

# 프로젝트 루트를 임포트 경로에 추가 (패키지가 아니라 스크립트 모음이라 필요)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_article(index: int) -> Dict[str, Any]:
    """코퍼스/쿼리 데이터 한 건. 실제 데이터와 같은 키를 쓴다."""
    return {
        "index": index,
        "news_title": f"원본 제목 {index}",
        "news_content": f"기사 본문 {index} " + "내용 " * 30,
        "new_title": f"사람이 쓴 낚시 제목 {index}",
    }


@pytest.fixture
def articles() -> List[Dict[str, Any]]:
    return [make_article(i) for i in range(10)]


@pytest.fixture
def article_file(tmp_path, articles):
    """기사 JSON 파일을 만들어 경로를 돌려주는 팩토리."""

    def _write(name: str = "articles.json", data: List[Dict[str, Any]] = None) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(data if data is not None else articles, ensure_ascii=False), encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def generated_dir(tmp_path):
    """judge_LLM 이 읽는 모델별 생성 결과 폴더를 만든다."""

    def _write(model_keys: List[str], count: int = 5, title_format: str = '{{"title": "{key} 제목 {i}"}}') -> str:
        directory = tmp_path / "generated"
        directory.mkdir(exist_ok=True)
        for key in model_keys:
            rows = [
                {
                    "index": i,
                    "query_title": f"원본 제목 {i}",
                    "query_content": f"기사 본문 {i}",
                    "human_direct": f"사람이 쓴 낚시 제목 {i}",
                    key: title_format.format(key=key, i=i),
                }
                for i in range(count)
            ]
            (directory / f"{key}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return str(directory)

    return _write


@pytest.fixture
def fake_judge(monkeypatch):
    """평가 모델 호출을 가짜로 바꾼다. 호출 순서대로 정해진 응답을 돌려준다."""
    import judge_LLM

    def _install(responses: List[str]):
        calls = {"count": 0, "messages": []}

        def _fake(model, messages, temperature, max_retries, **kwargs):
            calls["messages"].append(messages)
            response = responses[calls["count"] % len(responses)]
            calls["count"] += 1
            return response

        monkeypatch.setattr(judge_LLM, "completion_with_retry", _fake)
        return calls

    return _install
