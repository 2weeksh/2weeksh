"""aggregate: 심사자 선택 파싱, 승률 집계, 신뢰구간."""
from aggregate import aggregate, parse_choice, wilson_interval

LABELS = ["A", "B", "C", "D", "E"]


def make_result(index: int, choice: str, order=None):
    """평가 결과 한 건. shuffled_order 는 '제시 위치 -> 모델' 매핑이다."""
    return {
        "index": index,
        "evaluation": choice,
        "shuffled_order": order or {"A": "Direct_Human", "B": "GPT", "C": "RAG_GPT", "D": "Gemini", "E": "RAG_Gemini"},
        "shuffled_titles": {label: f"제목 {label}" for label in LABELS},
    }


class TestParseChoice:
    def test_정상_json에서_선택을_꺼낸다(self):
        assert parse_choice('{"choice": "B"}', LABELS) == "B"

    def test_코드펜스로_감싸도_파싱한다(self):
        assert parse_choice('```json\n{"choice": "C"}\n```', LABELS) == "C"

    def test_소문자를_대문자로_맞춘다(self):
        assert parse_choice('{"choice": "d"}', LABELS) == "D"

    def test_공백을_제거한다(self):
        assert parse_choice('{"choice": " E "}', LABELS) == "E"

    def test_후보에_없는_라벨은_버린다(self):
        assert parse_choice('{"choice": "Z"}', LABELS) == ""

    def test_json이_아니면_빈_문자열(self):
        assert parse_choice("저는 B를 선택합니다", LABELS) == ""

    def test_choice가_문자열이_아니면_빈_문자열(self):
        assert parse_choice('{"choice": 2}', LABELS) == ""


class TestWilsonInterval:
    def test_표본이_없으면_0으로_돌려준다(self):
        assert wilson_interval(0, 0) == {"lower": 0.0, "upper": 0.0}

    def test_구간이_실제_승률을_포함한다(self):
        interval = wilson_interval(50, 100)
        assert interval["lower"] < 0.5 < interval["upper"]

    def test_표본이_커지면_구간이_좁아진다(self):
        narrow = wilson_interval(500, 1000)
        wide = wilson_interval(5, 10)
        assert (narrow["upper"] - narrow["lower"]) < (wide["upper"] - wide["lower"])

    def test_0과_1을_벗어나지_않는다(self):
        assert wilson_interval(0, 5)["lower"] == 0.0
        assert wilson_interval(5, 5)["upper"] == 1.0


class TestAggregate:
    def test_선택을_제시위치가_아니라_모델에_귀속시킨다(self):
        """같은 'A' 선택이라도 셔플 결과에 따라 다른 모델의 승리가 된다."""
        results = [
            make_result(
                0,
                '{"choice": "A"}',
                {"A": "GPT", "B": "Gemini", "C": "RAG_GPT", "D": "RAG_Gemini", "E": "Direct_Human"},
            ),
            make_result(
                1,
                '{"choice": "A"}',
                {"A": "Gemini", "B": "GPT", "C": "RAG_GPT", "D": "RAG_Gemini", "E": "Direct_Human"},
            ),
        ]
        summary = aggregate(results)
        wins = {stat["model"]: stat["wins"] for stat in summary["model_stats"]}

        assert wins["GPT"] == 1
        assert wins["Gemini"] == 1

    def test_승률의_합이_1이_된다(self):
        summary = aggregate([make_result(i, '{"choice": "B"}') for i in range(4)])

        assert round(sum(stat["win_rate"] for stat in summary["model_stats"]), 4) == 1.0

    def test_파싱_실패는_유효_건수에서_빠진다(self):
        results = [make_result(0, '{"choice": "A"}'), make_result(1, "형식을 지키지 않은 응답")]
        summary = aggregate(results)

        assert summary["total_items"] == 2
        assert summary["valid_items"] == 1
        assert summary["invalid_items"] == 1

    def test_전부_실패해도_0으로_나누지_않는다(self):
        summary = aggregate([make_result(0, "평문"), make_result(1, "평문")])

        assert summary["valid_items"] == 0
        assert all(stat["win_rate"] == 0.0 for stat in summary["model_stats"])

    def test_제시_위치별_비율을_따로_집계한다(self):
        """순서 편향 확인용. 특정 위치만 뽑히면 셔플이 제 역할을 못 한 것이다."""
        summary = aggregate([make_result(i, '{"choice": "A"}') for i in range(5)])

        assert summary["position_stats"]["A"]["rate"] == 1.0
        assert summary["position_stats"]["B"]["rate"] == 0.0

    def test_등장_횟수는_모든_모델이_같다(self):
        summary = aggregate([make_result(i, '{"choice": "A"}') for i in range(3)])

        assert {stat["appearances"] for stat in summary["model_stats"]} == {3}
