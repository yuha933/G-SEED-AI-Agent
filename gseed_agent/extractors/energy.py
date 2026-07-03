from __future__ import annotations

import re
from typing import Any

from .common import make_fact, parse_number


# 에너지사용계획서가 직접 연결될 수 있는 G-SEED 항목
RELATED_CRITERIA = ["R-2.1", "R-2.2", "R-2.3", "R-2.4", "R-2.5", "R-ID-02"]


def extract(parsed_doc: dict[str, Any]) -> dict[str, Any]:
    """에너지사용계획서 OCR 텍스트를 G-SEED 산식 변수로 구조화한다."""
    facts: list[dict[str, Any]] = []

    for page in parsed_doc["pages"]:
        text = _normalize_text(page.get("text", ""))
        if not text:
            continue

        filename = parsed_doc["filename"]
        page_no = page["page"]
        confidence = 0.60 if page.get("ocr_used") else 0.80

        facts.extend(_extract_energy_performance_values(text, filename, page_no, confidence))
        facts.extend(_extract_energy_efficiency_grade(text, filename, page_no, confidence))
        facts.extend(_extract_green_home_saving_ratio(text, filename, page_no, confidence))
        facts.extend(_extract_renewable_ratio(text, filename, page_no, confidence))
        facts.extend(_extract_low_carbon_energy_source_values(text, filename, page_no, confidence))
        facts.extend(_extract_boolean_candidates(text, filename, page_no, confidence))

    return {"facts": _deduplicate(facts), "candidate_criteria": RELATED_CRITERIA}


def _extract_energy_performance_values(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """에너지성능지표 평점합계/EPI 점수 후보를 찾는다."""
    facts: list[dict[str, Any]] = []
    for line in _lines(text):
        if _is_reference_or_note_line(line):
            continue
        if not any(keyword in line for keyword in ["에너지성능지표", "성능지표", "EPI", "평점합계"]):
            continue
        value = _last_reasonable_number(line, min_value=0, max_value=120)
        if value is None:
            continue
        facts.append(
            make_fact(
                "energy_performance_index_total_score",
                value,
                "점",
                filename,
                page_no,
                line,
                confidence,
            )
        )
    return facts


def _extract_energy_efficiency_grade(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """건축물/건물 에너지효율등급을 G-SEED 변수로 추출한다."""
    facts: list[dict[str, Any]] = []
    keywords = ["건물에너지효율인증", "건물에너지 효율인증", "건물에너지 효율 인증", "건축물에너지효율등급", "에너지효율등급"]

    for keyword in keywords:
        for window in _windows_around(text, keyword, before=80, after=220):
            grade = _parse_energy_grade(window)
            if grade is None:
                continue

            if _is_planning_or_review_context(window) and not _is_confirmed_certification_context(window):
                facts.append(
                    make_fact(
                        "building_energy_efficiency_grade_candidate",
                        grade,
                        "등급",
                        filename,
                        page_no,
                        window,
                        max(confidence - 0.20, 0.30),
                        status="reference_only",
                    )
                )
                continue

            facts.append(
                make_fact(
                    "building_energy_efficiency_grade",
                    grade,
                    "등급",
                    filename,
                    page_no,
                    window,
                    confidence,
                )
            )
            facts.append(
                make_fact(
                    "building_energy_rating_mentioned",
                    True,
                    None,
                    filename,
                    page_no,
                    window,
                    confidence,
                )
            )
    return facts


def _extract_green_home_saving_ratio(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """친환경주택/에너지 절감률 후보를 찾는다."""
    facts: list[dict[str, Any]] = []
    for line in _lines(text):
        if _is_reference_or_note_line(line):
            continue
        if not any(keyword in line for keyword in ["친환경주택", "총 에너지", "총에너지", "에너지절약형"]):
            continue
        if not any(keyword in line for keyword in ["절감률", "절감율", "절감한", "절감"]):
            continue
        # LED 개별 설비 절감률처럼 G-SEED 2.1 전체 절감률이 아닌 값은 직접 점수화하지 않는다.
        if any(keyword in line for keyword in ["LED", "형광", "백열", "조명", "전력절감", "선로손실"]):
            continue
        value = _last_reasonable_number(line, min_value=0, max_value=100)
        if value is None:
            continue
        facts.append(
            make_fact(
                "green_home_energy_saving_ratio",
                value,
                "%",
                filename,
                page_no,
                line,
                confidence,
            )
        )
    return facts


def _extract_renewable_ratio(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """신재생에너지 설치비율 후보를 찾는다."""
    facts: list[dict[str, Any]] = []
    keywords = ["신재생에너지", "신 재 생", "신·재생에너지", "태양광발전", "태양열", "지열"]
    for line in _lines(text):
        if _is_reference_or_note_line(line):
            continue
        if not any(keyword in line for keyword in keywords):
            continue
        if "%" not in line and "비율" not in line and "설치비율" not in line:
            continue
        if "확대도입" in line and not re.search(r"\d+(?:\.\d+)?\s*%|％", line):
            continue
        value = _last_reasonable_number(line, min_value=0, max_value=100)
        if value is None:
            continue
        if _is_planning_or_review_context(line) and not _is_confirmed_application_context(line):
            facts.append(
                make_fact(
                    "renewable_energy_installation_ratio_candidate",
                    value,
                    "%",
                    filename,
                    page_no,
                    line,
                    max(confidence - 0.20, 0.30),
                    status="reference_only",
                )
            )
            continue
        facts.append(
            make_fact(
                "renewable_energy_installation_ratio",
                value,
                "%",
                filename,
                page_no,
                line,
                confidence,
            )
        )
    return facts


def _extract_low_carbon_energy_source_values(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """R-2.4 저탄소 에너지원 기술 관련 직접 변수를 찾는다."""
    facts: list[dict[str, Any]] = []

    for window in _windows_around(text, "지역난방", before=120, after=160):
        if not _is_project_specific_application_context(window):
            facts.append(
                make_fact(
                    "district_heating_building_candidate",
                    True,
                    None,
                    filename,
                    page_no,
                    window,
                    max(confidence - 0.20, 0.30),
                    status="reference_only",
                )
            )
            break
        facts.append(
            make_fact(
                "district_heating_building",
                True,
                None,
                filename,
                page_no,
                window,
                confidence,
            )
        )
        break

    for window in _windows_around(text, "지역냉방", before=120, after=160):
        if not _is_project_specific_application_context(window):
            facts.append(
                make_fact(
                    "district_cooling_building_candidate",
                    True,
                    None,
                    filename,
                    page_no,
                    window,
                    max(confidence - 0.20, 0.30),
                    status="reference_only",
                )
            )
            break
        facts.append(make_fact("district_cooling_building", True, None, filename, page_no, window, confidence))
        break

    if "열병합" in text and any(keyword in text for keyword in ["15%", "15 %", "난방", "급탕"]):
        context = _context(text, "열병합")
        if not _is_project_specific_application_context(context):
            facts.append(
                make_fact(
                    "cogeneration_heat_capacity_15_percent_designed_candidate",
                    True,
                    None,
                    filename,
                    page_no,
                    context,
                    max(confidence - 0.20, 0.30),
                    status="reference_only",
                )
            )
            return facts
        facts.append(
            make_fact(
                "cogeneration_heat_capacity_15_percent_designed",
                True,
                None,
                filename,
                page_no,
                context,
                confidence,
            )
        )

    return facts


def _extract_boolean_candidates(text: str, filename: str, page_no: int, confidence: float) -> list[dict[str, Any]]:
    """수치화 전 단계에서 의미 있는 적용/언급 후보를 보존한다."""
    specs = {
        "high_efficiency_equipment_mentioned": ["고효율에너지기자재", "고효율 기자재", "고효율인증기기", "고효율 인증"],
        "bems_mentioned": ["BEMS", "건물에너지관리시스템"],
        "led_lighting_applied": ["LED 도입", "LED등", "LED램프", "LED조명"],
        "standby_power_reduction_outlet_applied": ["대기전력저감형 콘센트", "에너지절감형 콘센트"],
        "daylight_applied": ["자연채광"],
        "high_efficiency_transport_equipment_applied": ["고효율 반송설비"],
    }
    facts: list[dict[str, Any]] = []
    for variable, keywords in specs.items():
        hit = next((keyword for keyword in keywords if keyword in text), None)
        if not hit:
            continue
        context = _context(text, hit)
        if _is_planning_or_review_context(context) and not _is_confirmed_application_context(context):
            facts.append(
                make_fact(
                    f"{variable}_candidate",
                    True,
                    None,
                    filename,
                    page_no,
                    context,
                    max(confidence - 0.20, 0.30),
                    status="reference_only",
                )
            )
            continue
        facts.append(make_fact(variable, True, None, filename, page_no, context, confidence))
    return facts


def _parse_energy_grade(text: str) -> str | None:
    """OCR 문장 안에서 1++/1+/1/2 등급을 안정적으로 읽는다."""
    normalized = text.replace("＋", "+").replace(" ", "")

    # 1++등급, 1+등급을 먼저 본다.
    match = re.search(r"1\+{1,2}\s*등급", normalized)
    if match:
        return match.group(0).replace("등급", "")

    # '2등급→1등급', '1등급 상향', '분양주택은 1등급' 같은 표현 대응
    grade_matches = re.findall(r"([1-5])\s*등급", text)
    if not grade_matches:
        return None

    if "상향" in text and "1" in grade_matches:
        return "1"
    if "받을 계획" in text and grade_matches:
        # 계획 문맥은 후보로만 저장되며, 실제 점수 변수로는 연결하지 않는다.
        return grade_matches[-1]
    if "2등급수1등급" in normalized or "2등급→1등급" in normalized:
        return "1"
    return grade_matches[-1]


def _normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("％", "%").replace("°/6", "%").replace("0/0", "%").replace("9/6", "%")
    return re.sub(r"[ \t]+", " ", text)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _windows_around(text: str, keyword: str, before: int = 80, after: int = 180) -> list[str]:
    windows: list[str] = []
    start = 0
    while True:
        idx = text.find(keyword, start)
        if idx < 0:
            break
        s = max(0, idx - before)
        e = min(len(text), idx + len(keyword) + after)
        windows.append(text[s:e])
        start = idx + len(keyword)
    return windows


def _last_reasonable_number(line: str, min_value: float, max_value: float) -> float | None:
    values = []
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", line):
        value = parse_number(raw)
        if value is None:
            continue
        if min_value <= value <= max_value:
            values.append(value)
    if not values:
        return None
    return values[-1]


def _is_reference_or_note_line(line: str) -> bool:
    compact = line.strip()
    if re.match(r"^(주\)?|주\d+\)|\d+\.)", compact):
        return True
    return any(word in compact for word in ["참조", "참고", "자료]", "산출근거", "재검토"])


def _is_planning_or_review_context(text: str) -> bool:
    """실제 적용/인증이 아니라 계획·검토·추진 단계로 보이는 문맥인지 판단한다."""
    compact = re.sub(r"\s+", "", text)
    planning_words = [
        "추진",
        "검토",
        "재검토",
        "확대도입",
        "도입검토",
        "방안제시",
        "받을계획",
        "상향추진",
        "계획서확인",
        "예정",
        "권장",
    ]
    return any(word in compact for word in planning_words)


def _is_confirmed_certification_context(text: str) -> bool:
    """에너지효율등급이 실제 인증/취득/예비인증 근거로 보이는지 판단한다."""
    compact = re.sub(r"\s+", "", text)
    confirmed_words = ["인증서", "인증결과", "인증등급", "예비인증", "본인증", "취득", "획득", "인증을받은", "등급을받은"]
    return any(word in compact for word in confirmed_words)


def _is_confirmed_application_context(text: str) -> bool:
    """설비/에너지원이 실제 적용 또는 사용 대상으로 보이는지 판단한다."""
    compact = re.sub(r"\s+", "", text)
    if _is_planning_or_review_context(compact) and not any(word in compact for word in ["적용", "사용", "설치"]):
        return False
    confirmed_words = ["적용", "사용", "설치", "적용함", "사용함", "설치함", "설치하여야한다", "적용하여야한다"]
    return any(word in compact for word in confirmed_words)


def _is_project_specific_application_context(text: str) -> bool:
    """일반 기술 설명표가 아니라 대상 건물 적용 문맥인지 조금 더 엄격하게 본다."""
    compact = re.sub(r"\s+", "", text)
    if not _is_confirmed_application_context(compact):
        return False
    generic_words = ["공공청사", "학교", "상업시설", "사업지구내도로", "설비명", "설치대상,장소", "하는시스템"]
    if any(word in compact for word in generic_words):
        return False
    project_words = ["신청주택", "주택조건", "공동주택", "아파트", "대상건축물", "본건물", "세대", "주거동", "해당건물"]
    return any(word in compact for word in project_words)


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
