from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import config


def build_gseed_kb(gseed_dir: str | Path = config.GSEED_DIR) -> dict[str, Any]:
    """G-SEED JSON DB를 챗봇/RAG용 계층형 Knowledge Base로 변환한다."""
    domains: list[dict[str, Any]] = []
    criteria_index: dict[str, dict[str, Any]] = {}

    for path in sorted(Path(gseed_dir).glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        domain = raw["전문분야"][0]
        domain_info = domain.get("전문분야", {})
        criteria_nodes = []

        for item in domain.get("인증항목", []):
            node = _criterion_node(item)
            criteria_nodes.append(node)
            criteria_index[node["criterion_id"]] = node

        domains.append(
            {
                "domain_no": domain_info.get("번호"),
                "domain_name": domain_info.get("명칭"),
                "source_file": path.name,
                "criteria": criteria_nodes,
            }
        )

    return {
        "schema_version": "1.0",
        "description": "G-SEED 2016-8 신축 주거용 건축물 기준 계층형 Knowledge Base",
        "domains": domains,
        "criteria_index": criteria_index,
    }


def save_gseed_kb(
    output_path: str | Path = config.DATA_DIR / "gseed_knowledge_base.json",
    gseed_dir: str | Path = config.GSEED_DIR,
) -> Path:
    """계층형 KB를 JSON 파일로 저장한다."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    kb = build_gseed_kb(gseed_dir)
    output.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_or_build_gseed_kb(output_path: str | Path = config.DATA_DIR / "gseed_knowledge_base.json") -> dict[str, Any]:
    """저장된 KB가 있으면 읽고, 없으면 새로 생성한다."""
    path = Path(output_path)
    if not path.exists():
        save_gseed_kb(path)
    return json.loads(path.read_text(encoding="utf-8"))


def search_kb(kb: dict[str, Any], query: str, limit: int = 5) -> list[dict[str, Any]]:
    """간단한 키워드 기반 검색.

    항목 번호나 항목명이 명확히 들어온 경우에는 해당 항목을 최우선으로 둔다.
    """
    normalized_query = query.strip().lower()
    query_terms = [term for term in re.split(r"\s+", normalized_query) if term]
    if not query_terms:
        return []

    criteria = list(kb.get("criteria_index", {}).values())
    exact_matches: list[dict[str, Any]] = []

    # R-6.3처럼 항목 번호를 직접 물은 경우.
    id_match = re.search(r"r[-\s]?(\d+)\.(\d+)", normalized_query)
    if id_match:
        target_id = f"R-{id_match.group(1)}.{id_match.group(2)}"
        criterion = kb.get("criteria_index", {}).get(target_id)
        if criterion:
            exact_matches.append(criterion)

    # 생태면적률처럼 항목명이 그대로 들어온 경우.
    compact_query = re.sub(r"\s+", "", normalized_query)
    for criterion in criteria:
        name = str(criterion.get("name") or "").lower()
        compact_name = re.sub(r"\s+", "", name)
        if compact_name and compact_name in compact_query and criterion not in exact_matches:
            exact_matches.append(criterion)

    scored: list[tuple[int, dict[str, Any]]] = []
    for criterion in criteria:
        if criterion in exact_matches:
            continue
        haystack = json.dumps(criterion, ensure_ascii=False).lower()
        score = sum(haystack.count(term) for term in query_terms)
        if score:
            scored.append((score, criterion))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = exact_matches + [criterion for _, criterion in scored]
    return results[:limit]


def _criterion_node(item: dict[str, Any]) -> dict[str, Any]:
    cert = item.get("인증항목", {})
    evaluation = item.get("평가", {})
    cm = evaluation.get("calculation_model", {})

    return {
        "criterion_id": item.get("criterion_id"),
        "name": cert.get("명칭"),
        "domain": item.get("전문분야", {}),
        "source": item.get("출처", {}),
        "score": cm.get("max_score") or cert.get("배점"),
        "purpose": evaluation.get("평가목적"),
        "method": evaluation.get("평가방법"),
        "score_formula": cm.get("score_formula"),
        "input_variables": [
            {
                "name": variable.get("name"),
                "label": variable.get("label"),
                "type": variable.get("type"),
                "unit": variable.get("unit"),
            }
            for variable in cm.get("input_variables", [])
        ],
        "derived_variables": cm.get("derived_variables", []),
        "grade_rules": _rules(cm),
        "point_items": cm.get("point_items", []),
        "required_documents": item.get("제출서류", {}),
        "references": item.get("참고자료", []),
        "raw_text": item.get("원문", ""),
    }


def _rules(cm: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ["grade_rules", "overall_grade_rules", "performance_grade_rules", "unit_grade_rules"]:
        if cm.get(key):
            return cm[key]
    return []
