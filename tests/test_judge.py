"""judge_LLM: 생성 결과 병합, 제목 파싱, 평가 루프와 이어하기."""
import json

import pytest

import judge_LLM
from judge_LLM import KEY_TO_LABEL_MAP, evaluation, load_generated, parse_title

MODEL_KEYS = [key for key in KEY_TO_LABEL_MAP if key != "human_direct"]


class TestParseTitle:
    def test_정상_json에서_제목을_꺼낸다(self):
        assert parse_title('{"title": "낚시성 제목"}') == ("낚시성 제목", True)

    def test_코드펜스로_감싸도_제목만_꺼낸다(self):
        # 실패하면 후보 제목 자리에 JSON 덩어리가 그대로 들어가 평가가 오염된다
        assert parse_title('```json\n{"title": "낚시성 제목"}\n```') == ("낚시성 제목", True)

    def test_제목_앞뒤_공백을_제거한다(self):
        assert parse_title('{"title": "  제목  "}') == ("제목", True)

    def test_파싱에_실패하면_원문과_실패표시를_돌려준다(self):
        assert parse_title("그냥 평문 제목") == ("그냥 평문 제목", False)

    def test_title_키가_없으면_실패로_본다(self):
        title, parsed = parse_title('{"headline": "제목"}')
        assert parsed is False

    def test_title이_빈_문자열이면_실패로_본다(self):
        title, parsed = parse_title('{"title": "   "}')
        assert parsed is False


class TestLoadGenerated:
    def test_모델별_파일을_index로_합친다(self, generated_dir):
        merged = load_generated(generated_dir(MODEL_KEYS, count=3))

        assert len(merged) == 3
        assert all(key in merged[0] for key in MODEL_KEYS)
        assert merged[0]["human_direct"] == "사람이 쓴 낚시 제목 0"

    def test_생성_건수가_다르면_불완전한_기사를_제외한다(self, tmp_path, generated_dir):
        directory = generated_dir(MODEL_KEYS, count=5)
        # 한 모델만 3건까지 생성하다 중단된 상황
        path = f"{directory}/{MODEL_KEYS[-1]}.json"
        rows = json.loads(open(path, encoding="utf-8").read())[:3]
        open(path, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False))

        merged = load_generated(directory)

        assert len(merged) == 3  # 5건이 아니라 모두 갖춰진 3건만

    def test_생성_파일이_없으면_선행_단계를_안내한다(self, generated_dir):
        directory = generated_dir(MODEL_KEYS[:-1], count=3)  # 한 모델 파일 자체가 없음

        with pytest.raises(SystemExit) as error:
            load_generated(directory)

        assert "generated_model.py" in str(error.value)


class TestEvaluation:
    def _run(self, directory, output_path, fake_judge, responses):
        judge_LLM.random.seed(42)
        calls = fake_judge(responses)
        results = evaluation(
            load_generated(directory),
            evaluate_model="fake/model",
            output_path=output_path,
            temperature=0.0,
            max_retries=1,
            save_every=100,
        )
        return results, calls

    def test_기사마다_한_번씩_평가한다(self, tmp_path, generated_dir, fake_judge):
        directory = generated_dir(MODEL_KEYS, count=5)
        results, calls = self._run(directory, str(tmp_path / "eval.json"), fake_judge, ['{"choice": "A"}'])

        assert len(results) == 5
        assert calls["count"] == 5

    def test_후보_5개가_모두_프롬프트에_들어간다(self, tmp_path, generated_dir, fake_judge):
        directory = generated_dir(MODEL_KEYS, count=1)
        _, calls = self._run(directory, str(tmp_path / "eval.json"), fake_judge, ['{"choice": "A"}'])

        prompt = calls["messages"][0][1]["content"]
        for label in ["A:", "B:", "C:", "D:", "E:"]:
            assert label in prompt

    def test_shuffled_order에_5개_모델이_모두_담긴다(self, tmp_path, generated_dir, fake_judge):
        directory = generated_dir(MODEL_KEYS, count=3)
        results, _ = self._run(directory, str(tmp_path / "eval.json"), fake_judge, ['{"choice": "A"}'])

        for result in results:
            assert sorted(result["shuffled_order"].values()) == sorted(judge_LLM.LABEL_TO_MODEL_MAP.values())

    def test_이어하기해도_후보_제시_순서가_동일하다(self, tmp_path, generated_dir, fake_judge):
        """중단 후 재실행 시 순서가 바뀌면 앞뒤 결과를 같은 실험으로 합칠 수 없다."""
        directory = generated_dir(MODEL_KEYS, count=6)

        full_path = str(tmp_path / "full.json")
        full, _ = self._run(directory, full_path, fake_judge, ['{"choice": "A"}'])

        # 앞 2건만 남기고 중단된 상태를 만든 뒤 이어서 실행
        resumed_path = str(tmp_path / "resumed.json")
        open(resumed_path, "w", encoding="utf-8").write(json.dumps(full[:2], ensure_ascii=False))
        resumed, calls = self._run(directory, resumed_path, fake_judge, ['{"choice": "A"}'])

        assert calls["count"] == 4  # 남은 4건만 호출
        assert [r["shuffled_order"] for r in resumed] == [r["shuffled_order"] for r in full]

    def test_코드펜스_응답이어도_후보_제목에_JSON이_섞이지_않는다(self, tmp_path, generated_dir, fake_judge):
        directory = generated_dir(MODEL_KEYS, count=2, title_format='```json\n{{"title": "{key} 제목 {i}"}}\n```')
        results, _ = self._run(directory, str(tmp_path / "eval.json"), fake_judge, ['{"choice": "A"}'])

        for title in results[0]["shuffled_titles"].values():
            assert "```" not in title
            assert '"title"' not in title

    def test_이어하기시_건너뛴_기사는_파싱실패로_세지_않는다(self, tmp_path, generated_dir, fake_judge, capsys):
        directory = generated_dir(MODEL_KEYS, count=4, title_format="평문 제목 {i}")  # 전부 파싱 실패

        full_path = str(tmp_path / "full.json")
        full, _ = self._run(directory, full_path, fake_judge, ['{"choice": "A"}'])
        capsys.readouterr()

        resumed_path = str(tmp_path / "resumed.json")
        open(resumed_path, "w", encoding="utf-8").write(json.dumps(full[:3], ensure_ascii=False))
        self._run(directory, resumed_path, fake_judge, ['{"choice": "A"}'])

        # 남은 1건 x 모델 4개 = 4건만 경고에 잡혀야 한다 (수정 전에는 16건)
        assert f"파싱 실패 {len(MODEL_KEYS)}건" in capsys.readouterr().out
