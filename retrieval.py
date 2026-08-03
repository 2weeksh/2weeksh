import argparse
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from bm25s import BM25
from kiwipiepy import Kiwi

from io_utils import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="유사 기사 검색하는 스크립트")
    # 경로 설정
    parser.add_argument("--clickbait_path", type=str, default="data/TL_Part1_Clickbait_Direct_merged.json")
    parser.add_argument("--non_clickbait_path", type=str, default="data/VL_Part1_Clickbait_Direct_merged.json")
    parser.add_argument("--rag_retrieval_path", type=str, default="outputs/rag_retrieval_results.json")
    # 하이퍼파라미터 설정
    parser.add_argument("--min_num_character", type=int, default=100, help="corpus에 포함될 기사 최소 글자 수")
    parser.add_argument("--max_num_character", type=int, default=3000, help="corpus에 포함될 기사 최대 글자 수")
    parser.add_argument("--k", type=int, default=5, help="검색할 유사 기사 개수")
    parser.add_argument("--sample_size", type=int, default=200, help="쿼리 샘플 개수")
    return parser


# 형태소 분석기 로딩이 무거워서 실제로 토큰화할 때 한 번만 초기화한다
_kiwi: Optional[Kiwi] = None


def get_tokenizer() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


# 토크나이저
def tokenize(texts: List[str]) -> List[List[str]]:
    kiwi = get_tokenizer()
    return [[token.form for token in kiwi.tokenize(text)] for text in texts]


# 검색 코퍼스 생성
def build_corpus(clickbait_path: str, min_len: int, max_len: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    corpus = []
    corpus_raw = []

    # 데이터 열기
    with open(clickbait_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    for item in corpus_data:
        text = f"{item['news_title']} {item['new_title']} {item['news_content']}"
        if min_len <= len(text) <= max_len:  # 글자 수 필터링
            # 본문이 같은 기사가 섞여 있어도 유실되지 않도록 위치를 맞춘 리스트로 관리
            corpus.append(text)
            corpus_raw.append(item)

    return corpus, corpus_raw


def create_retriever(corpus: List[str]) -> BM25:
    # corpus 토큰화 및 BM25 인덱싱 (corpus를 넘기지 않으면 retrieve가 문서 인덱스를 반환)
    corpus_tokens = tokenize(corpus)
    retriever = BM25(k1=1.5, b=0.75)
    retriever.index(corpus_tokens)
    return retriever


def retrieve_title(
    non_clickbait_path: str, retriever: BM25, corpus_raw: List[Dict[str, Any]], k: int, sample_size: int
) -> List[Dict[str, Any]]:
    # 데이터 열기
    with open(non_clickbait_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    # 코퍼스/쿼리 개수보다 큰 값이 들어오면 bm25s와 random.sample이 예외를 내므로 미리 맞춰준다
    if k > len(corpus_raw):
        print(f"[경고] k({k})가 corpus 크기({len(corpus_raw)})보다 커서 {len(corpus_raw)}로 줄입니다.")
        k = len(corpus_raw)
    if sample_size > len(query_data):
        print(f"[경고] sample_size({sample_size})가 쿼리 개수({len(query_data)})보다 커서 전체를 사용합니다.")
        sample_size = len(query_data)

    sampled_queries = random.sample(query_data, sample_size)

    rag_queries = []

    # 기사마다 검색 수행
    for query in sampled_queries:
        query_text = f"{query['news_title']} {query['news_content']}"
        query_tokens = tokenize([query_text])[0]

        # 검색
        doc_index_list, _ = retriever.retrieve([query_tokens], k=k)
        retrieved_articles = []

        # 검색 문서들 저장
        for doc_index in doc_index_list[0]:
            raw_doc = corpus_raw[int(doc_index)]
            retrieved_articles.append(
                {
                    "non_clickbait_title": raw_doc["news_title"],
                    "clickbait_title": raw_doc["new_title"],
                    "clickbait_content": raw_doc["news_content"],
                }
            )

        # 결과 양식
        rag_queries.append(
            {
                "query_title": query["news_title"],
                "query_content": query["news_content"],
                "human_direct_clickbait_title": query["new_title"],
                "retrieved_articles": retrieved_articles,
            }
        )

    return rag_queries


def main() -> None:
    args = build_parser().parse_args()
    random.seed(42)

    corpus_list, corpus_map = build_corpus(args.clickbait_path, args.min_num_character, args.max_num_character)

    # 코퍼스가 비면 BM25 인덱싱이 내부에서 알아보기 힘든 예외로 죽으므로 여기서 막는다
    if not corpus_list:
        raise SystemExit(
            f"[오류] 길이 조건({args.min_num_character}~{args.max_num_character}자)을 만족하는 기사가 없습니다.\n"
            f"       입력 경로와 필터 값을 확인하세요: {args.clickbait_path}"
        )

    print(f"corpus {len(corpus_list)}건 구축 완료")

    bm25_retriever = create_retriever(corpus_list)
    rag_queries = retrieve_title(args.non_clickbait_path, bm25_retriever, corpus_map, args.k, args.sample_size)

    # 저장
    save_json(args.rag_retrieval_path, rag_queries)
    print(f"{len(rag_queries)}건 저장 -> {args.rag_retrieval_path}")


if __name__ == "__main__":
    main()
