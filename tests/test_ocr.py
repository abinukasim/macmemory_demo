from app.utils.ocr import _normalize_ocr_text


def test_normalize_ocr_text_strips_noise() -> None:
    assert _normalize_ocr_text("  Holiday   market \n hours\tfrom 10am to 8pm daily  ") == "Holiday market hours from 10am to 8pm daily"


def test_normalize_ocr_text_rejects_short_text() -> None:
    assert _normalize_ocr_text("menu") == ""
