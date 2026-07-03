from __future__ import annotations

import re
from typing import Any


def make_fact(
    variable: str,
    value: Any,
    unit: str | None,
    source_document: str,
    source_page: int,
    source_text: str,
    confidence: float = 0.7,
    status: str = "needs_review",
) -> dict[str, Any]:
    """설계값 후보를 공통 형식으로 만든다."""
    return {
        "variable": variable,
        "value": value,
        "unit": unit,
        "source_document": source_document,
        "source_page": source_page,
        "source_text": source_text[:300],
        "confidence": confidence,
        "status": status,
    }


def parse_number(text: str) -> float | None:
    """문자열에서 숫자를 추출한다."""
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return None
    return float(m.group(0).replace(",", ""))

