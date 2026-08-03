"""io_utils: JSON 저장/체크포인트/LLM 응답 파싱."""
import json
import os

from io_utils import load_checkpoint, parse_json_object, save_json


class TestParseJsonObject:
    """모델이 JSON 형식을 지키지 않는 경우가 잦아서, 실패 형태별로 확인한다."""

    def test_평범한_json을_그대로_파싱한다(self):
        assert parse_json_object('{"title": "제목"}') == {"title": "제목"}

    def test_코드펜스로_감싼_json도_파싱한다(self):
        # 수정 전에는 여기서 실패해 원문 전체가 제목으로 쓰였다
        assert parse_json_object('```json\n{"title": "제목"}\n```') == {"title": "제목"}

    def test_언어표시_없는_코드펜스도_파싱한다(self):
        assert parse_json_object('```\n{"title": "제목"}\n```') == {"title": "제목"}

    def test_앞뒤에_설명이_붙어도_파싱한다(self):
        assert parse_json_object('물론이죠! {"title": "제목"} 도움이 되셨길') == {"title": "제목"}

    def test_dict가_들어오면_그대로_돌려준다(self):
        assert parse_json_object({"title": "제목"}) == {"title": "제목"}

    def test_json이_아니면_None을_돌려준다(self):
        assert parse_json_object("그냥 평문 제목") is None

    def test_문자열이_아니면_None을_돌려준다(self):
        assert parse_json_object(None) is None
        assert parse_json_object(123) is None

    def test_객체가_아닌_json은_None을_돌려준다(self):
        assert parse_json_object("[1, 2, 3]") is None


class TestSaveJson:
    def test_상위_폴더가_없으면_만들어_저장한다(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "result.json")
        save_json(path, {"key": "값"})

        assert json.loads(open(path, encoding="utf-8").read()) == {"key": "값"}

    def test_임시파일을_남기지_않는다(self, tmp_path):
        path = str(tmp_path / "result.json")
        save_json(path, [1, 2, 3])

        assert not os.path.exists(f"{path}.tmp")

    def test_한글이_이스케이프되지_않는다(self, tmp_path):
        path = str(tmp_path / "result.json")
        save_json(path, {"제목": "낚시성"})

        assert "낚시성" in open(path, encoding="utf-8").read()

    def test_기존_파일을_덮어쓴다(self, tmp_path):
        path = str(tmp_path / "result.json")
        save_json(path, {"v": 1})
        save_json(path, {"v": 2})

        assert json.loads(open(path, encoding="utf-8").read()) == {"v": 2}


class TestLoadCheckpoint:
    def test_파일이_없으면_빈_리스트(self, tmp_path):
        assert load_checkpoint(str(tmp_path / "없는파일.json")) == []

    def test_저장한_내용을_그대로_읽는다(self, tmp_path):
        path = str(tmp_path / "ckpt.json")
        save_json(path, [{"index": 0}])

        assert load_checkpoint(path) == [{"index": 0}]

    def test_파일이_깨져_있으면_빈_리스트로_넘어간다(self, tmp_path):
        # 저장 도중 중단돼 잘린 파일이 있어도 전체 실행이 죽지 않아야 한다
        path = tmp_path / "broken.json"
        path.write_text('[{"index": 0}, {"ind', encoding="utf-8")

        assert load_checkpoint(str(path)) == []
