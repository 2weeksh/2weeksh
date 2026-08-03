"""retrieval: 코퍼스 구축과 BM25 검색.

bm25s / kiwipiepy 가 설치돼 있어야 실행된다. (Windows 등 미설치 환경에서는 건너뜀)
형태소 분석기는 느려서 공백 분리 토크나이저로 대체한다.
"""
import pytest

pytest.importorskip("bm25s", reason="bm25s 미설치")
pytest.importorskip("kiwipiepy", reason="kiwipiepy 미설치")

import retrieval  # noqa: E402
from retrieval import build_corpus, create_retriever, retrieve_title  # noqa: E402


@pytest.fixture(autouse=True)
def fast_tokenizer(monkeypatch):
    """Kiwi 로딩 없이 돌도록 공백 분리로 대체한다."""
    monkeypatch.setattr(retrieval, "tokenize", lambda texts: [text.split() for text in texts])


class TestBuildCorpus:
    def test_제목과_본문을_합쳐_코퍼스를_만든다(self, article_file, articles):
        corpus, corpus_raw = build_corpus(article_file(), min_len=0, max_len=10000)

        assert len(corpus) == len(articles)
        assert articles[0]["new_title"] in corpus[0]  # 낚시 제목도 검색 대상에 포함

    def test_코퍼스와_원본이_같은_순서로_짝지어진다(self, article_file, articles):
        """검색 결과 인덱스로 원본을 되찾으므로 순서가 어긋나면 안 된다."""
        corpus, corpus_raw = build_corpus(article_file(), min_len=0, max_len=10000)

        for text, raw in zip(corpus, corpus_raw):
            assert raw["news_title"] in text

    def test_너무_짧은_기사를_걸러낸다(self, article_file, articles):
        short = dict(articles[0], news_content="짧음", news_title="짧", new_title="짧")
        corpus, _ = build_corpus(article_file("short.json", [short] + articles), min_len=100, max_len=10000)

        assert len(corpus) == len(articles)

    def test_너무_긴_기사를_걸러낸다(self, article_file, articles):
        long_article = dict(articles[0], news_content="긴 본문 " * 2000)
        corpus, _ = build_corpus(article_file("long.json", [long_article] + articles), min_len=0, max_len=3000)

        assert len(corpus) == len(articles)

    def test_조건에_맞는_기사가_없으면_빈_결과(self, article_file):
        corpus, corpus_raw = build_corpus(article_file(), min_len=999999, max_len=1000000)

        assert corpus == [] and corpus_raw == []


class TestRetrieveTitle:
    def _retriever(self, article_file, articles):
        corpus, corpus_raw = build_corpus(article_file(), min_len=0, max_len=10000)
        return create_retriever(corpus), corpus_raw

    def test_쿼리마다_k건씩_검색한다(self, article_file, articles):
        retriever, corpus_raw = self._retriever(article_file, articles)
        results = retrieve_title(article_file("query.json"), retriever, corpus_raw, k=3, sample_size=4)

        assert len(results) == 4
        assert all(len(r["retrieved_articles"]) == 3 for r in results)

    def test_검색_결과에_사람이_쓴_낚시제목이_담긴다(self, article_file, articles):
        retriever, corpus_raw = self._retriever(article_file, articles)
        results = retrieve_title(article_file("query.json"), retriever, corpus_raw, k=2, sample_size=2)

        clickbait_titles = [a["clickbait_title"] for a in results[0]["retrieved_articles"]]
        assert all(title.startswith("사람이 쓴 낚시 제목") for title in clickbait_titles)

    def test_결과_양식에_필요한_키가_모두_있다(self, article_file, articles):
        retriever, corpus_raw = self._retriever(article_file, articles)
        results = retrieve_title(article_file("query.json"), retriever, corpus_raw, k=1, sample_size=1)

        assert set(results[0]) == {
            "query_title",
            "query_content",
            "human_direct_clickbait_title",
            "retrieved_articles",
        }

    def test_k가_코퍼스보다_크면_코퍼스_크기로_줄인다(self, article_file, articles, capsys):
        retriever, corpus_raw = self._retriever(article_file, articles)
        results = retrieve_title(article_file("query.json"), retriever, corpus_raw, k=999, sample_size=2)

        assert len(results[0]["retrieved_articles"]) == len(corpus_raw)
        assert "[경고]" in capsys.readouterr().out

    def test_sample_size가_쿼리보다_크면_전체를_사용한다(self, article_file, articles, capsys):
        retriever, corpus_raw = self._retriever(article_file, articles)
        results = retrieve_title(article_file("query.json"), retriever, corpus_raw, k=2, sample_size=999)

        assert len(results) == len(articles)
        assert "[경고]" in capsys.readouterr().out

    def test_같은_시드면_같은_쿼리가_뽑힌다(self, article_file, articles):
        retriever, corpus_raw = self._retriever(article_file, articles)
        query_path = article_file("query.json")

        retrieval.random.seed(42)
        first = retrieve_title(query_path, retriever, corpus_raw, k=1, sample_size=3)
        retrieval.random.seed(42)
        second = retrieve_title(query_path, retriever, corpus_raw, k=1, sample_size=3)

        assert [r["query_title"] for r in first] == [r["query_title"] for r in second]
