from __future__ import annotations

import re
from typing import Any

from .common import make_fact


RELATED_CRITERIA = ["R-3.1", "R-3.2", "R-3.3", "R-3.4", "R-7.1"]


def extract(parsed_doc: dict[str, Any]) -> dict[str, Any]:
    """마감자재 목록표에서 자재 후보를 구조화한다.

    이 문서는 제품 목록일 뿐 인증서/성적서가 아니므로 EPD, 저탄소, 저방출 점수를 확정하지 않는다.
    """
    all_text = "\n".join(p.get("text", "") for p in parsed_doc["pages"])
    filename = parsed_doc["filename"]
    facts: list[dict[str, Any]] = []

    products = _extract_product_like_lines(all_text)
    manufacturers = _extract_manufacturers(all_text)
    low_emission_candidates = _count_keywords(
        all_text,
        ["실크벽지", "도배지", "장판", "강마루", "도장", "시트", "PET", "LPM", "MDF", "PB", "접착", "몰딩"],
    )
    recycled_or_resource_candidates = _count_keywords(all_text, ["재활용", "재생", "순환", "폐", "리사이클"])
    hazardous_reduction_candidates = _count_keywords(all_text, ["친환경", "무독성", "무석면", "저VOC", "TVOC", "폼알데하이드", "포름알데히드"])
    epd_candidates = _count_keywords(all_text, ["환경성선언", "EPD", "환경표지", "저탄소제품", "탄소발자국"])

    facts.append(
        make_fact(
            "material_product_candidate_count",
            len(products),
            "개",
            filename,
            1,
            f"제품명/마감유형으로 보이는 후보 {len(products)}개",
            confidence=0.65,
            status="needs_review",
        )
    )
    facts.append(
        make_fact(
            "material_manufacturer_candidate_count",
            len(manufacturers),
            "개",
            filename,
            1,
            f"제조사 후보: {', '.join(sorted(manufacturers)[:20])}",
            confidence=0.60,
            status="needs_review",
        )
    )
    facts.append(
        make_fact(
            "low_emission_material_candidate_count",
            low_emission_candidates,
            "개",
            filename,
            1,
            "실내공기 오염물질 저방출 성적서 확인이 필요한 마감재 후보 수",
            confidence=0.45,
            status="needs_review",
        )
    )

    # 아래 인증계수는 키워드가 문서에 실제로 있을 때만 후보로 둔다. 없으면 점수 산정 변수로 넘기지 않는다.
    if epd_candidates:
        facts.append(_candidate_fact("epd_product_total_count", epd_candidates, filename, "환경성선언/EPD 관련 키워드 후보"))
    if recycled_or_resource_candidates:
        facts.append(_candidate_fact("resource_circulation_material_count", recycled_or_resource_candidates, filename, "자원순환 관련 키워드 후보"))
    if hazardous_reduction_candidates:
        facts.append(_candidate_fact("hazardous_substance_reduction_material_count", hazardous_reduction_candidates, filename, "유해물질 저감 관련 키워드 후보"))

    return {"facts": facts, "candidate_criteria": RELATED_CRITERIA}


def _candidate_fact(variable: str, value: int, filename: str, source_text: str) -> dict[str, Any]:
    return make_fact(variable, value, "개", filename, 1, source_text, confidence=0.40, status="needs_review")


def _extract_product_like_lines(text: str) -> list[str]:
    products: list[str] = []
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        if len(line) < 2:
            continue
        if line in {"실명", "구분", "마감유형", "제품명", "규격", "제조사명", "이미지", "비고"}:
            continue
        if re.search(r"[A-Z]{1,5}[-\d][A-Z0-9\-]*", line) or any(k in line for k in ["벽지", "타일", "LED", "도어", "장판", "강마루", "대리석", "수전"]):
            products.append(line)
    return products


def _extract_manufacturers(text: str) -> set[str]:
    known = {
        "LG하우시스",
        "KCC",
        "삼영크란츠",
        "DID벽지",
        "서울벽지",
        "신한벽지",
        "개나리벽지",
        "IS동서",
        "미존테크",
        "현대코리안",
        "르그랑코리아",
        "남양조명",
        "하나룩스",
        "경동원",
        "아메리칸스탠다드",
        "다로스",
        "하츠",
        "삼성",
        "LG",
    }
    return {name for name in known if name in text}


def _count_keywords(text: str, keywords: list[str]) -> int:
    return sum(text.count(keyword) for keyword in keywords)
