from __future__ import annotations

from typing import Any

from .common import make_fact


RELATED_CRITERIA = ["R-5.1", "R-ID-63"]


def extract(parsed_doc: dict[str, Any]) -> dict[str, Any]:
    """시공계획서에서 현장 환경관리 관련 설계값 후보를 추출한다."""
    facts: list[dict[str, Any]] = []
    all_text = "\n".join(p.get("text", "") for p in parsed_doc["pages"])
    filename = parsed_doc["filename"]

    def add(variable: str, value: Any, source_text: str, confidence: float = 0.65) -> None:
        facts.append(
            make_fact(
                variable,
                value,
                None,
                filename,
                _find_page(parsed_doc, source_text),
                source_text,
                confidence=confidence,
                status="needs_review",
            )
        )

    has_environment_plan = _has_any(all_text, ["환경관리계획", "환경 관리 계획", "환경관리"])
    has_organization = _has_any(all_text, ["조직표", "담당조직", "조직"])
    has_implementation = _has_any(all_text, ["분리수거", "청소", "정리정돈", "폐기물", "환경법규", "환경 보전 활동"])

    if has_environment_plan:
        add("site_environmental_management_plan_document_exists", True, _context(all_text, "환경관리계획"))
        add("site_environmental_management_plan_established", True, _context(all_text, "환경관리계획"))
    if has_organization:
        add("site_environmental_management_organization_exists", True, _context(all_text, "조직표"))
    if has_implementation:
        add("site_environmental_management_implemented", True, _context(all_text, "환경관리계획"))

    # 혁신항목 ID-63은 수행범위 7개 항목 전체에 대한 보고/모니터링 성격이 필요하다.
    # 현재 시공계획서에서는 일부 환경관리 활동만 확인되므로 count 후보로만 둔다.
    item_hits = {
        "waste_management": ["폐기물", "분리수거", "쓰레기"],
        "cleaning_housekeeping": ["청소", "정리정돈", "청결"],
        "environment_law_compliance": ["환경법규", "환경 보전"],
        "environment_monitoring_report": ["모니터링", "리포트", "보고서"],
        "noise_dust_control": ["소음", "분진", "비산먼지"],
        "water_pollution_control": ["오수", "폐수", "수질"],
        "traffic_ecology_protection": ["교통", "생태", "보전"],
    }
    count = sum(1 for words in item_hits.values() if _has_any(all_text, words))
    if count:
        add("construction_environmental_management_item_count", count, "환경관리 수행범위 후보 항목 수", confidence=0.55)

    if _has_any(all_text, ["모니터링", "리포트", "보고서"]) and count >= 3:
        add(
            "construction_environmental_management_report_including_items_1_2_3_exists",
            True,
            "환경관리 모니터링/보고서 및 주요 항목 포함 후보",
            confidence=0.50,
        )

    return {"facts": facts, "candidate_criteria": RELATED_CRITERIA}


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _context(text: str, keyword: str, width: int = 300) -> str:
    idx = text.find(keyword)
    if idx < 0:
        return text[:width]
    start = max(0, idx - 120)
    return text[start : start + width]


def _find_page(parsed_doc: dict[str, Any], source_text: str) -> int:
    token = source_text[:30].strip()
    for page in parsed_doc["pages"]:
        if token and token in page.get("text", ""):
            return page["page"]
    return 1
