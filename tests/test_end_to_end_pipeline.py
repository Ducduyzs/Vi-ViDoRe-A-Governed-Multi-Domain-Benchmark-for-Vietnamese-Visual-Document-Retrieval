import pytest
from pathlib import Path
from PIL import Image

from src.data.schema import PageMetadata, QueryItem, QrelItem, BenchmarkSplit, DomainType, PageType, DocumentSourceType, QueryType
from src.models.text_baselines import BM25Retriever
from src.evaluation.evaluator import ViViDoReEvaluator
from src.evaluation.report_generator import generate_markdown_report, generate_latex_table

def test_end_to_end_pipeline_flow(tmp_path: Path):
    # 1. Mock page metadata
    pages = [
        PageMetadata(
            doc_id="legal_doc_01",
            page_num=1,
            page_id="legal_doc_01_p001",
            file_path=str(tmp_path / "doc01.pdf"),
            image_path=str(tmp_path / "doc01_p001.png"),
            sha256="sha_01",
            phash="phash_01",
            domain=DomainType.LEGAL,
            page_type=PageType.TEXT_HEAVY,
            source_type=DocumentSourceType.BORN_DIGITAL,
            native_text="Quy định về xử phạt vi phạm hành chính trong lĩnh vực bảo hiểm xã hội.",
        ),
        PageMetadata(
            doc_id="fin_doc_02",
            page_num=1,
            page_id="fin_doc_02_p001",
            file_path=str(tmp_path / "doc02.pdf"),
            image_path=str(tmp_path / "doc02_p001.png"),
            sha256="sha_02",
            phash="phash_02",
            domain=DomainType.FINANCIAL,
            page_type=PageType.TABLE_HEAVY,
            source_type=DocumentSourceType.BORN_DIGITAL,
            native_text="Báo cáo tài chính quý 3 năm 2023 doanh thu thuần đạt 500 tỷ đồng.",
        ),
    ]

    # 2. Mock queries & qrels
    queries = [
        QueryItem(
            query_id="q_1",
            query_text="Mức phạt vi phạm bảo hiểm xã hội?",
            domain=DomainType.LEGAL,
            query_type=QueryType.LEGAL_CLAUSE,
            source="human_written",
            target_page_ids=["legal_doc_01_p001"],
        ),
        QueryItem(
            query_id="q_2",
            query_text="Doanh thu thuần quý 3 năm 2023?",
            domain=DomainType.FINANCIAL,
            query_type=QueryType.NUMERIC_TABLE,
            source="llm_assisted",
            target_page_ids=["fin_doc_02_p001"],
        ),
    ]

    qrels = [
        QrelItem(query_id="q_1", page_id="legal_doc_01_p001", relevance=2),
        QrelItem(query_id="q_2", page_id="fin_doc_02_p001", relevance=2),
    ]

    split = BenchmarkSplit(
        name="test",
        queries=queries,
        corpus_page_ids=["legal_doc_01_p001", "fin_doc_02_p001"],
        qrels=qrels,
        doc_sources=["legal_doc_01", "fin_doc_02"],
    )

    # 3. Test BM25 retrieval
    bm25 = BM25Retriever()
    corpus_texts = [p.native_text for p in pages]
    bm25.index_corpus(split.corpus_page_ids, corpus_texts)

    results = bm25.retrieve(
        queries=[q.query_text for q in queries],
        query_ids=[q.query_id for q in queries],
        top_k=2,
    )

    # 4. Test Evaluator
    evaluator = ViViDoReEvaluator(split, pages_metadata=pages)
    metrics = evaluator.evaluate_retrieval_results(results, model_name="BM25 Baseline")

    assert metrics["num_queries"] == 2
    assert "overall" in metrics
    assert "ndcg@5" in metrics["overall"]
    assert metrics["macro_domain_ndcg@5"] > 0.0

    # 5. Test Reports
    md_report = generate_markdown_report([metrics])
    latex_table = generate_latex_table([metrics])

    assert "BM25 Baseline" in md_report
    assert "begin{table*}" in latex_table

