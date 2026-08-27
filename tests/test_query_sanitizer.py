import pytest
from src.data.query_sanitizer import QuerySanitizer

def test_deictic_detection():
    sanitizer = QuerySanitizer()

    bad_query_1 = "Theo như bảng này thì doanh thu quý 3 là bao nhiêu?"
    deictic = sanitizer.check_deictic_words(bad_query_1)
    assert len(deictic) > 0

    bad_query_2 = "Hình trên thể hiện quy trình cấp phép gì?"
    deictic2 = sanitizer.check_deictic_words(bad_query_2)
    assert len(deictic2) > 0

    good_query = "Quy trình cấp giấy chứng nhận quyền sử dụng đất bao gồm những bước nào?"
    deictic_good = sanitizer.check_deictic_words(good_query)
    assert len(deictic_good) == 0

def test_clean_deictic():
    sanitizer = QuerySanitizer()
    query = "Cho biết mức phạt theo như bảng này đối với xe máy vượt đèn đỏ?"
    cleaned = sanitizer.clean_deictic_words(query)
    assert "bảng này" not in cleaned
    assert "mức phạt" in cleaned

def test_lexical_leakage():
    sanitizer = QuerySanitizer(max_ngram_leakage=4)
    doc_text = "Hành vi không đội mũ bảo hiểm bị xử phạt tiền từ hai trăm nghìn đồng đến ba trăm nghìn đồng theo quy định hiện hành."
    leaked_query = "không đội mũ bảo hiểm bị xử phạt tiền từ hai trăm nghìn đồng"
    leakage = sanitizer.compute_lexical_leakage(leaked_query, doc_text)
    assert leakage >= 0.7

    natural_query = "Mức xử phạt đối với người điều khiển xe gắn máy không đội mũ bảo hiểm?"
    leakage_natural = sanitizer.compute_lexical_leakage(natural_query, doc_text)
    assert leakage_natural < 0.6

