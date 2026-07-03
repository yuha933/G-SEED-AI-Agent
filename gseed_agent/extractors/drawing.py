from __future__ import annotations

import re
from typing import Any

from .common import make_fact, parse_number


# 현재 건축도면 PDF는 여러 예시 도면이 섞여 있으므로,
# 사용자가 지정했던 공동주택 실험 범위(6~31페이지)만 우선 신뢰한다.
TARGET_PAGE_START = 6
TARGET_PAGE_END = 31


def extract(parsed_doc: dict[str, Any]) -> dict[str, Any]:
    """건축도면에서 G-SEED 점수 산정에 바로 쓸 수 있는 값만 추출한다."""
    facts: list[dict[str, Any]] = []
    candidate_criteria: set[str] = set()

    for page in parsed_doc["pages"]:
        page_no = int(page.get("page") or 0)
        if not (TARGET_PAGE_START <= page_no <= TARGET_PAGE_END):
            continue

        text = _normalize_text(page.get("text", ""))
        if not text:
            continue

        filename = parsed_doc["filename"]
        confidence = 0.55 if page.get("ocr_used") else 0.78

        page_facts = []
        page_facts.extend(_extract_design_overview(text, filename, page_no, confidence))
        page_facts.extend(_extract_unit_ventilation(text, filename, page_no, confidence))
        page_facts.extend(_extract_heating_control(text, filename, page_no, confidence))
        page_facts.extend(_extract_reference_only_candidates(text, filename, page_no, confidence))

        for fact in page_facts:
            facts.append(fact)
            if fact["variable"] in {
                "unit_ventilation_rate_0_5_ach_available",
                "health_friendly_housing_ventilation_equipment_installed",
                "unit_indoor_pollutant_measurement_and_airflow_auto_control",
                "room_indoor_pollutant_measurement_and_airflow_auto_control",
            }:
                candidate_criteria.add("R-7.3")
            if fact["variable"] in {
                "household_heating_batch_on_off_control_available",
                "household_heating_batch_timer_control_available",
                "room_heating_temperature_control_with_valves_available",
                "room_heating_timer_temperature_control_available",
            }:
                candidate_criteria.add("R-7.4")

    return {
        "facts": _deduplicate(facts),
        "candidate_criteria": sorted(candidate_criteria),
    }


def _extract_design_overview(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """설계개요 표에서 건물 기본값을 추출한다.

    이 값들은 후속 항목 매핑 후보로 보존하되,
    해당 항목을 단독 평가할 근거가 부족하면 candidate_criteria에는 넣지 않는다.
    """
    facts: list[dict[str, Any]] = []
    if "설계 개요" not in text and "설계개요" not in text:
        return facts

    specs = [
        ("site_area_m2", "대지 면적", "m2", _site_area_from_design_overview),
        ("building_area_m2", "건축 면적", "m2", _building_area_from_design_overview),
        ("total_households", "총 세대수", "세대", _total_households_from_design_overview),
    ]
    for variable, label, unit, parser in specs:
        value = parser(text)
        if value is None:
            continue
        facts.append(
            make_fact(
                variable,
                value,
                unit,
                filename,
                page_no,
                _context(text, label),
                confidence,
                status="needs_review",
            )
        )

    parking_count = _planned_parking_count(text)
    if parking_count is not None:
        facts.append(
            make_fact(
                "planned_parking_space_count",
                parking_count,
                "대",
                filename,
                page_no,
                _context(text, "주 차 대 수"),
                confidence,
                status="needs_review",
            )
        )

    return facts


def _extract_unit_ventilation(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """단위세대 환기성능(R-7.3) 근거를 찾는다."""
    if not any(keyword in text for keyword in ["환기덕트", "환 기 유 니 트", "필요환기량", "환기횟수"]):
        return []

    facts: list[dict[str, Any]] = []

    # G-SEED R-7.3의 4급 기준은 세대 내 시간당 0.5회 환기가 가능한지 여부다.
    if re.search(r"환기횟수[\s\S]{0,180}0\.5", text) or "0.5" in _context(text, "환기횟수", width=300):
        facts.append(
            make_fact(
                "unit_ventilation_rate_0_5_ach_available",
                True,
                None,
                filename,
                page_no,
                _context(text, "환기횟수", width=360),
                confidence,
                status="needs_review",
            )
        )

    # 고효율/KS 인증 환기유니트는 성능 근거 후보로만 보존한다.
    # '건강친화형 주택 기준' 문구가 없으면 상위 등급 변수로 확정하지 않는다.
    if "건강친화형" in text and any(keyword in text for keyword in ["환기유니트", "환 기 유 니 트", "환기설비"]):
        facts.append(
            make_fact(
                "health_friendly_housing_ventilation_equipment_installed",
                True,
                None,
                filename,
                page_no,
                _context(text, "건강친화형"),
                confidence,
                status="needs_review",
            )
        )

    return facts


def _extract_heating_control(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """자동온도조절장치 설치 수준(R-7.4) 근거를 찾는다."""
    facts: list[dict[str, Any]] = []

    if "실별자동온도조절장치" in text or ("실별" in text and "온도조절" in text):
        facts.append(
            make_fact(
                "room_heating_temperature_control_with_valves_available",
                True,
                None,
                filename,
                page_no,
                _context(text, "실별자동온도조절장치" if "실별자동온도조절장치" in text else "온도조절"),
                confidence,
                status="needs_review",
            )
        )

    if "시간예약" in text and "온도" in text:
        facts.append(
            make_fact(
                "room_heating_timer_temperature_control_available",
                True,
                None,
                filename,
                page_no,
                _context(text, "시간예약"),
                confidence,
                status="needs_review",
            )
        )

    if "일괄" in text and "난방" in text and ("ON/OFF" in text.upper() or "ON" in text.upper()):
        facts.append(
            make_fact(
                "household_heating_batch_on_off_control_available",
                True,
                None,
                filename,
                page_no,
                _context(text, "일괄"),
                confidence,
                status="needs_review",
            )
        )

    return facts


def _extract_reference_only_candidates(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """점수 확정에는 부족하지만 보고서에 보여줄 참고 설계값을 보존한다."""
    specs = {
        "led_lighting_applied": ["LED 조명", "LED조명", "LED 50W", "LED 40W"],
        "standby_power_reduction_outlet_applied": ["대기전력차단스위치", "대기전력저감"],
        "lighting_batch_off_switch_installed": ["일괄소등스위치"],
        "pressure_reducing_valve_mentioned": ["감압 V/V", "감압밸브"],
        "remote_water_meter_mentioned": ["원격검침"],
    }
    facts: list[dict[str, Any]] = []
    for variable, keywords in specs.items():
        hit = next((keyword for keyword in keywords if keyword in text), None)
        if not hit:
            continue
        facts.append(
            make_fact(
                variable,
                True,
                None,
                filename,
                page_no,
                _context(text, hit),
                confidence,
                status="needs_review",
            )
        )
    return facts


def _number_near_label(text: str, label: str) -> float | None:
    idx = text.find(label)
    if idx < 0:
        return None
    window = text[idx : idx + 260]
    return parse_number(window.replace(label, "", 1))


def _site_area_from_design_overview(text: str) -> float | None:
    """설계개요 표의 대지면적을 표 순서 기반으로 읽는다."""
    match = re.search(
        r"제3종일반주거지역[^\n]*\n\s*([\d,]+(?:\.\d+)?)\s*㎡\s*\n\s*0\.0000\s*㎡\s*\n\s*([\d,]+(?:\.\d+)?)\s*㎡",
        text,
    )
    if match:
        return parse_number(match.group(1))
    return _number_near_label(text, "대지 면적")


def _building_area_from_design_overview(text: str) -> float | None:
    """설계개요 표의 건축면적을 구조/규모 행 다음 값으로 읽는다."""
    match = re.search(r"철근콘크리트[^\n]*\n\s*([\d,]+(?:\.\d+)?)\s*㎡", text)
    if match:
        return parse_number(match.group(1))
    return _number_near_label(text, "건축 면적")


def _total_households_from_design_overview(text: str) -> float | None:
    """총 세대수는 '총 세대수' 라벨 인접값을 우선한다."""
    match = re.search(r"총\s*세대수\s*\n?\s*([\d,]+)", text)
    if match:
        return parse_number(match.group(1))
    return _number_near_label(text, "세 대 수")


def _planned_parking_count(text: str) -> float | None:
    idx = text.find("주 차 대 수")
    if idx < 0:
        idx = text.find("주차 대수")
    if idx < 0:
        return None

    window = text[idx : idx + 900]
    match = re.search(r"합\s*계\s*\n?\s*([\d,]+)\s*대", window)
    if not match:
        match = re.search(r"합\s*계\s*\n\s*([\d,]+)\s*대\s*\n\s*100\.00%", text)
    if not match:
        return None
    return parse_number(match.group(1))


def _normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text)


def _context(text: str, keyword: str, width: int = 260) -> str:
    idx = text.find(keyword)
    if idx < 0:
        return text[:width]
    start = max(0, idx - 100)
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
