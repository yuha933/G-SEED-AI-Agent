from __future__ import annotations

import re
from typing import Any

from .common import make_fact, parse_number


RELATED_CRITERIA = ["R-1.1", "R-4.1", "R-6.1", "R-6.2", "R-6.3", "R-6.4", "R-ID-06"]


def extract(parsed_doc: dict[str, Any]) -> dict[str, Any]:
    """조경설계도에서 조경·생태 관련 설계값 후보를 추출한다."""
    facts: list[dict[str, Any]] = []

    for page in parsed_doc["pages"]:
        text = _normalize_text(page.get("text", ""))
        if not text:
            continue

        confidence = 0.50 if page.get("ocr_used") else 0.70
        filename = parsed_doc["filename"]
        page_no = page["page"]

        facts.extend(
            _extract_keyword_numbers(
                text=text,
                filename=filename,
                page_no=page_no,
                confidence=confidence,
                specs=[
                    ("site_area_m2", ["대지면적", "대지 면적"], "m2"),
                    ("landscape_area_m2", ["조경면적", "조경 면적"], "m2"),
                    ("landscape_area_ratio", ["조경면적률", "조경율", "조경률"], "%"),
                    ("green_area_m2", ["녹지면적", "녹지 면적"], "m2"),
                    ("natural_ground_green_area_m2", ["자연지반녹지", "자연 지반 녹지", "자연지반"], "m2"),
                    ("ecological_area_ratio", ["생태면적률", "생태 면적률"], "%"),
                    ("ecological_area_m2", ["생태면적", "생태 면적"], "m2"),
                    ("reused_topsoil_volume_m3", ["표토재활용", "표토 재활용", "재활용 표토"], "m3"),
                    ("permeable_pavement_area_m2", ["투수포장", "투수 포장"], "m2"),
                    ("rainwater_facility_capacity_m3", ["빗물이용시설", "우수저류", "우수 저류", "빗물저류"], "m3"),
                    ("biotope_area_m2", ["비오톱", "수생비오톱", "육생비오톱"], "m2"),
                ],
            )
        )

        # 수목 수량표는 면적이 아니라 개수라 별도 변수로 둔다.
        facts.extend(
            _extract_keyword_numbers(
                text=text,
                filename=filename,
                page_no=page_no,
                confidence=confidence,
                specs=[
                    ("tree_count_tall", ["교목", "상록교목", "낙엽교목"], "주"),
                    ("tree_count_shrub", ["관목", "상록관목", "낙엽관목"], "주"),
                    ("groundcover_area_m2", ["초화", "지피", "지피식물"], "m2"),
                ],
            )
        )

        for variable, keywords in {
            "biotope_mentioned": ["비오톱", "수생비오톱", "육생비오톱"],
            "native_species_mentioned": ["자생종", "향토수종", "지역 자생"],
            "ecological_network_mentioned": ["생태축", "녹지축", "생태 네트워크"],
            "rainwater_facility_mentioned": ["빗물이용시설", "우수저류", "빗물저류"],
        }.items():
            hit = next((keyword for keyword in keywords if keyword in text), None)
            if hit:
                facts.append(make_fact(variable, True, None, filename, page_no, _context(text, hit), confidence))

    return {"facts": _deduplicate(facts), "candidate_criteria": RELATED_CRITERIA}


def _normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r", "\n"))


def _extract_keyword_numbers(
    text: str,
    filename: str,
    page_no: int,
    confidence: float,
    specs: list[tuple[str, list[str], str]],
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for variable, keywords, unit in specs:
        for line in lines:
            if not any(keyword in line for keyword in keywords):
                continue
            value = _number_after_keywords(line, keywords)
            if value is None:
                value = parse_number(line)
            if value is None:
                continue
            facts.append(make_fact(variable, value, unit, filename, page_no, line, confidence))
            break
    return facts


def _number_after_keywords(line: str, keywords: list[str]) -> float | None:
    for keyword in keywords:
        idx = line.find(keyword)
        if idx < 0:
            continue
        tail = line[idx + len(keyword) :]
        value = parse_number(tail)
        if value is not None:
            return value
    return None


def _context(text: str, keyword: str, width: int = 220) -> str:
    idx = text.find(keyword)
    if idx < 0:
        return text[:width]
    start = max(0, idx - 80)
    return text[start : start + width]


def _deduplicate(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, Any, int]] = set()
    result: list[dict[str, Any]] = []
    for fact in facts:
        key = (fact["variable"], fact["value"], fact["source_page"])
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result
