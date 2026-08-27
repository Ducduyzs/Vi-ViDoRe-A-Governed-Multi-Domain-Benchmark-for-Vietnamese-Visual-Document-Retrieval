import pytest
from src.data.schema import PageMetadata, QueryItem, QrelItem, DomainType, PageType, DocumentSourceType, QueryType

def test_page_metadata_serialization():
    meta = PageMetadata(
        doc_id="legal_doc_01",
        page_num=1,
        page_id="legal_doc_01_p001",
        file_path="/data/doc01.pdf",
        image_path="/data/pages/legal_doc_01_p001.png",
        sha256="abcdef123456",
        phash="10101010",
        domain=DomainType.LEGAL,
        page_type=PageType.TEXT_HEAVY,
        source_type=DocumentSourceType.BORN_DIGITAL,
        native_text="Cộng hòa Xã hội Chủ nghĩa Việt Nam",
        char_count=35,
    )
    d = meta.to_dict()
    assert d["domain"] == "legal"
    assert d["page_type"] == "text_heavy"
    assert d["source_type"] == "born_digital"

    reconstructed = PageMetadata.from_dict(d)
    assert reconstructed.doc_id == meta.doc_id
    assert reconstructed.domain == DomainType.LEGAL

def test_query_item_serialization():
    q = QueryItem(
        query_id="q_01",
        query_text="Mức phạt vi phạm quy định về bảo hiểm xã hội?",
        domain=DomainType.LEGAL,
        query_type=QueryType.LEGAL_CLAUSE,
        source="human_written",
        target_page_ids=["legal_doc_01_p001"],
    )
    d = q.to_dict()
    assert d["query_type"] == "legal_clause"
    reconstructed = QueryItem.from_dict(d)
    assert reconstructed.query_text == q.query_text
    assert reconstructed.query_type == QueryType.LEGAL_CLAUSE

def test_qrel_tsv_conversion():
    row = "q_01\t0\tlegal_doc_01_p001\t2"
    qrel = QrelItem.from_tsv_row(row)
    assert qrel.query_id == "q_01"
    assert qrel.page_id == "legal_doc_01_p001"
    assert qrel.relevance == 2
    assert qrel.to_tsv_row() == row

