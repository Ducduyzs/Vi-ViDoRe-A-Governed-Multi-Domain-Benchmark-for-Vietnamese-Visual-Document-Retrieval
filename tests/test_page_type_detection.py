from pathlib import Path

from PIL import Image, ImageDraw

from scripts.detect_page_types import classify_page, repair_mojibake


def _image(tmp_path: Path, name: str, grid: bool = False) -> Path:
    path = tmp_path / name
    image = Image.new("L", (600, 800), "white")
    if grid:
        draw = ImageDraw.Draw(image)
        for y in range(100, 701, 60):
            draw.line((50, y, 550, y), fill="black", width=3)
        for x in range(50, 551, 100):
            draw.line((x, 100, x, 700), fill="black", width=3)
    image.save(path)
    return path


def test_scanned_page_is_ranked_as_scanned(tmp_path):
    page = {
        "native_text": "", "char_count": 0, "source_type": "scanned",
        "image_path": str(_image(tmp_path, "scan.png")),
    }
    scores, _, _ = classify_page(page)
    assert scores["scanned"] >= 0.9


def test_table_signals_score_table_candidate(tmp_path):
    page = {
        "native_text": "Bảng 2 Đơn vị tính: triệu đồng\n2023   120   350   470\n2024   140   390   530",
        "char_count": 86, "source_type": "born_digital",
        "image_path": str(_image(tmp_path, "table.png", grid=True)),
    }
    scores, evidence, _ = classify_page(page)
    assert scores["table_heavy"] >= 0.35
    assert evidence["table_keywords"]


def test_form_keywords_score_form_candidate(tmp_path):
    page = {
        "native_text": "MẪU SỐ 01 - ĐƠN ĐĂNG KÝ\nHọ và tên: ..........\nNgày sinh: ..........\nKý tên",
        "char_count": 80, "source_type": "born_digital",
        "image_path": str(_image(tmp_path, "form.png")),
    }
    scores, evidence, _ = classify_page(page)
    assert scores["form_or_template"] >= 0.5
    assert evidence["form_keywords"]


def test_repairs_common_utf8_latin1_mojibake():
    assert "Biểu đồ" in repair_mojibake("Biá»ƒu Ä‘á»“")
