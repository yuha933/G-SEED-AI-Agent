from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any


class GSeedChecker:
    """G-SEED 기준 로딩, 변수 매핑, 판정을 한 파일에서 처리한다."""

    def __init__(self, gseed_dir: str | Path) -> None:
        self.gseed_dir = Path(gseed_dir)
        self.criteria = self._load_criteria()
        self.total_reference_score = self._calc_total_reference_score()

    def _load_criteria(self) -> dict[str, dict[str, Any]]:
        criteria: dict[str, dict[str, Any]] = {}
        for path in sorted(self.gseed_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            domain = list(data.values())[1][0]
            items = domain[list(domain.keys())[-1]]
            for item in items:
                criteria[item["criterion_id"]] = item
        return criteria

    def _criterion_max_score(self, criterion: dict[str, Any]) -> float:
        cert = criterion.get("인증항목", {})
        cm = criterion.get("평가", {}).get("calculation_model", {})
        try:
            return float(cm.get("max_score") or cert.get("배점") or 0)
        except Exception:
            return 0.0

    def _calc_total_reference_score(self) -> float:
        """전체 기준 DB의 원점수 총합을 계산한다."""
        return sum(self._criterion_max_score(c) for c in self.criteria.values())

    def evaluate(
        self,
        facts: list[dict[str, Any]],
        candidate_criteria: list[str] | None,
        target_score: float,
    ) -> dict[str, Any]:
        """후보 항목을 평가하고 점수 요약을 만든다."""
        fact_map = self._build_fact_map(facts)
        criteria_ids = candidate_criteria or list(self.criteria.keys())
        results = [self._evaluate_one(self.criteria[cid], fact_map) for cid in criteria_ids if cid in self.criteria]

        raw_estimated_score = sum(r.get("score") or 0 for r in results if r["status"] in {"pass", "needs_review"})
        evaluated_max_score = sum(r.get("max_score") or 0 for r in results)
        normalized_score = self._normalize_score(raw_estimated_score)
        score_gap = max(0.0, target_score - normalized_score)
        status_counts = self._status_counts(results)
        coverage_summary = self._coverage_summary(results)

        return {
            "summary": {
                "estimated_score": round(normalized_score, 2),
                "score_basis": "100점 환산",
                "raw_estimated_score": round(raw_estimated_score, 2),
                "total_reference_score": round(self.total_reference_score, 2),
                "evaluated_max_score": round(evaluated_max_score, 2),
                "target_score": target_score,
                "score_gap": round(score_gap, 2),
                "evaluated_criteria_count": len(results),
            },
            "status_counts": status_counts,
            "coverage_summary": coverage_summary,
            "criteria_results": results,
            "recommendation_sets": self._recommend(results, score_gap),
            "missing_documents": self._missing_documents(results),
        }

    def _evaluate_one(self, criterion: dict[str, Any], fact_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
        cm = criterion.get("평가", {}).get("calculation_model", {})
        cert = criterion.get("인증항목", {})
        max_score = self._criterion_max_score(criterion)

        context = {k: v["value"] for k, v in fact_map.items()}
        context.update(self._derive_values(cm, context))
        rules = self._get_rules(cm)

        missing_variables: set[str] = set()
        review_variables: set[str] = set()
        used_variable_names: set[str] = set()
        unparsed_rule_count = 0

        for rule in rules:
            condition = rule.get("condition", "")
            names = self._names_in_expr(condition)
            if condition and not names:
                unparsed_rule_count += 1
                continue
            used_variable_names.update(names)
            missing = [n for n in names if n not in context and n not in {"and", "or", "not", "in"}]
            if missing:
                missing_variables.update(missing)

            # OR 조건은 일부 변수만 있어도 판정 가능한 경우가 있다.
            # 예: EPI 점수는 없지만 에너지효율등급이 1등급이면 에너지 성능 판정 가능.
            eval_context = self._context_with_missing_defaults(condition, context, missing)

            if self._safe_eval(condition, eval_context):
                weight = rule.get("weight")
                score = self._score_from_rule(weight, max_score, cm, rule)
                if "energy_performance_score" in context:
                    score = context["energy_performance_score"]
                evidence = [fact_map[n] for n in names if n in fact_map]
                if not evidence:
                    evidence = self._fallback_evidence_from_inputs(cm, fact_map)
                for n in names:
                    if fact_map.get(n, {}).get("status") == "needs_review":
                        review_variables.add(n)
                for item in evidence:
                    if item.get("status") == "needs_review":
                        review_variables.add(item.get("variable", ""))
                return {
                    "criterion_id": criterion["criterion_id"],
                    "name": cert.get("명칭"),
                    "status": "needs_review" if review_variables else "pass",
                    "grade": rule.get("grade"),
                    "score": round(score, 2) if score is not None else None,
                    "max_score": max_score,
                    "evidence": evidence,
                    "missing_variables": [],
                    "reason": "기준을 만족했으나 일부 근거값 검토가 필요합니다." if review_variables else "기준을 만족합니다.",
                }

        if missing_variables:
            required_variables = self._source_required_variables(cm, missing_variables, context)
            missing_for_debug = self._debug_missing_variables(required_variables, fact_map)
            return {
                "criterion_id": criterion["criterion_id"],
                "name": cert.get("명칭"),
                "status": "insufficient_evidence",
                "grade": None,
                "score": None,
                "max_score": max_score,
                "evidence": self._fallback_evidence_from_inputs(cm, fact_map),
                "missing_variables": missing_for_debug,
                "required_variables": required_variables,
                "reason": "현재 문서에서 일부 단서는 있으나, 등급 산정에 필요한 핵심 근거가 부족합니다.",
            }

        # 근거값이 없거나 조건식을 해석하지 못한 경우에는 미달로 단정하지 않는다.
        if not used_variable_names or unparsed_rule_count == len(rules):
            return {
                "criterion_id": criterion["criterion_id"],
                "name": cert.get("명칭"),
                "status": "not_evaluated",
                "grade": None,
                "score": None,
                "max_score": max_score,
                "evidence": [],
                "missing_variables": [],
                "required_variables": self._input_variable_names(cm),
                "reason": "현재 문서 범위에서는 이 항목을 판단하지 않았습니다.",
            }

        return {
            "criterion_id": criterion["criterion_id"],
            "name": cert.get("명칭"),
            "status": "fail",
            "grade": None,
            "score": 0,
            "max_score": max_score,
            "evidence": list(fact_map.values()),
            "missing_variables": [],
            "reason": "확인된 값 기준으로 등급 조건을 만족하지 못했습니다.",
        }

    def _build_fact_map(self, facts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        # 같은 변수 후보가 여러 개면 confidence가 높은 값을 사용한다.
        result: dict[str, dict[str, Any]] = {}
        for fact in facts:
            if fact.get("status") == "reference_only":
                continue
            var = fact.get("variable")
            if not var:
                continue
            if var not in result or fact.get("confidence", 0) > result[var].get("confidence", 0):
                result[var] = fact
        return result

    def _fallback_evidence_from_inputs(self, cm: dict[str, Any], fact_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """파생변수로 판정된 경우 원 입력 변수 근거를 evidence로 연결한다."""
        input_names = [item.get("name") for item in cm.get("input_variables", []) if item.get("name")]
        return [fact_map[name] for name in input_names if name in fact_map]

    def _input_variable_names(self, cm: dict[str, Any]) -> list[str]:
        """calculation_model의 직접 입력 변수명을 반환한다."""
        return [item.get("name") for item in cm.get("input_variables", []) if item.get("name")]

    def _source_required_variables(
        self,
        cm: dict[str, Any],
        missing_variables: set[str],
        context: dict[str, Any],
    ) -> list[str]:
        """파생변수 missing을 사용자가 이해할 수 있는 원본 필요 변수로 풀어낸다."""
        input_names = set(self._input_variable_names(cm))
        derived_formulas = {
            item.get("name"): item.get("formula", "")
            for item in cm.get("derived_variables", [])
            if item.get("name")
        }
        point_formulas = {
            item.get("name"): item.get("condition", "")
            for item in cm.get("point_items", [])
            if item.get("name")
        }
        formula_map = {**derived_formulas, **point_formulas}

        required: set[str] = set()

        def expand(name: str, depth: int = 0) -> None:
            if depth > 4:
                required.add(name)
                return
            if name in input_names:
                if name not in context:
                    required.add(name)
                return
            formula = formula_map.get(name)
            if not formula:
                required.add(name)
                return
            names = self._names_in_expr(formula)
            if not names:
                required.add(name)
                return
            for child in names:
                if child in self._allowed_eval_names():
                    continue
                if child not in context:
                    expand(child, depth + 1)

        for variable in missing_variables:
            expand(variable)

        # 원본 변수를 하나도 특정하지 못하면, 해당 항목의 직접 입력 변수 중 아직 없는 값을 보여준다.
        if not required:
            required = {name for name in input_names if name not in context}

        return sorted(required)

    def _debug_missing_variables(
        self,
        required_variables: list[str],
        fact_map: dict[str, dict[str, Any]],
    ) -> list[str]:
        """진짜 추출 실패 의심 변수만 missing_variables에 남긴다.

        지금은 문서 단서가 있는 같은 항목의 입력값이 일부라도 잡힌 경우에만
        missing으로 표시하고, 일반적인 문서 부재는 required_variables로만 둔다.
        """
        if not fact_map:
            return []
        known_names = set(fact_map)
        if not known_names.intersection(required_variables):
            return []
        return [name for name in required_variables if name not in fact_map]

    def _get_rules(self, cm: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ["grade_rules", "overall_grade_rules", "performance_grade_rules", "unit_grade_rules"]:
            if cm.get(key):
                return cm[key]
        return []

    def _derive_values(self, cm: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        derived: dict[str, Any] = {}
        derived.update(self._derive_rule_weights(cm, context))
        derived.update(self._derive_point_items(cm, {**context, **derived}))
        for item in cm.get("derived_variables", []):
            name = item.get("name")
            formula = item.get("formula", "")
            if not name or "for each" in formula or "look up" in formula:
                continue
            eval_context = {**context, **derived}
            names = self._names_in_expr(formula)
            missing = [n for n in names if n not in eval_context and n not in self._allowed_eval_names()]
            if missing and not any(n in eval_context for n in names):
                continue
            eval_context = self._context_with_missing_defaults(formula, eval_context, missing)
            value = self._safe_eval(formula, eval_context)
            if value is not None:
                derived[name] = value
        return derived

    def _derive_point_items(self, cm: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """점수 항목표(point_items)를 부분 근거 기반으로 계산한다."""
        derived: dict[str, Any] = {}
        for item in cm.get("point_items", []):
            name = item.get("name")
            condition = item.get("condition", "")
            if not name or not condition:
                continue
            names = self._names_in_expr(condition)
            missing = [n for n in names if n not in context and n not in self._allowed_eval_names()]
            if missing:
                continue
            derived[name] = float(item.get("point", 0)) if self._safe_eval(condition, context) else 0
        return derived

    def _derive_rule_weights(self, cm: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """보조 등급표의 condition을 이용해 weight_by_* 변수를 만든다."""
        derived: dict[str, Any] = {}
        mapping = {
            "method_2_grade_rules": "weight_by_building_energy_efficiency_grade",
            "method_3_grade_rules": "weight_by_green_home_energy_saving_ratio",
        }
        for rule_key, weight_name in mapping.items():
            for rule in cm.get(rule_key, []):
                condition = rule.get("condition", "")
                names = self._names_in_expr(condition)
                missing = [n for n in names if n not in context]
                eval_context = self._context_with_missing_defaults(condition, context, missing)
                if self._safe_eval(condition, eval_context):
                    derived[weight_name] = float(rule.get("weight", 0))
                    break
        return derived

    def _score_from_rule(self, weight: Any, max_score: float, cm: dict[str, Any], rule: dict[str, Any]) -> float | None:
        if weight is not None and max_score:
            return float(weight) * max_score
        # 점수형 항목은 grade에 숫자가 들어가는 경우가 있다.
        grade = str(rule.get("grade", ""))
        m = re.search(r"(\d+(?:\.\d+)?)", grade)
        if m:
            return float(m.group(1))
        if "score = 1 if" in str(cm.get("score_formula", "")):
            return float(max_score or 1)
        return None

    def _normalize_score(self, raw_score: float) -> float:
        """원점수를 100점 기준으로 환산한다."""
        if self.total_reference_score <= 0:
            return raw_score
        return raw_score / self.total_reference_score * 100

    def _context_with_missing_defaults(
        self,
        expr: str,
        context: dict[str, Any],
        missing: list[str],
    ) -> dict[str, Any]:
        """일부 변수 누락 시에도 OR 조건을 평가할 수 있게 보수적 기본값을 채운다."""
        patched = dict(context)
        for name in missing:
            if name in patched:
                continue
            if re.search(rf"{re.escape(name)}\s+in\s+\[", expr):
                patched[name] = None
            elif re.search(rf"{re.escape(name)}\s*==", expr):
                patched[name] = None
            else:
                patched[name] = 0
        return patched

    def _safe_eval(self, expr: str, context: dict[str, Any]) -> Any:
        if not expr:
            return None
        allowed = {
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "ceil": math.ceil,
            "atan": math.atan,
            "degrees": math.degrees,
            "math": math,
        }
        try:
            return eval(expr, {"__builtins__": {}, **allowed}, context)
        except Exception:
            return None

    def _allowed_eval_names(self) -> set[str]:
        return {"abs", "min", "max", "round", "ceil", "atan", "degrees", "math"}

    def _names_in_expr(self, expr: str) -> set[str]:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return set()
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def _status_counts(self, results: list[dict[str, Any]]) -> dict[str, int]:
        statuses = ["pass", "fail", "needs_review", "insufficient_evidence", "not_evaluated", "unknown"]
        return {s: sum(1 for r in results if r["status"] == s) for s in statuses}

    def _coverage_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """점수보다 문서 검토 커버리지를 설명하기 위한 요약값."""
        evaluable = [r for r in results if r["status"] in {"pass", "fail", "needs_review"}]
        needs_more_evidence = [r for r in results if r["status"] == "insufficient_evidence"]
        extraction_suspects = [r for r in results if r.get("missing_variables")]
        return {
            "evaluated_count": len(evaluable),
            "needs_review_count": sum(1 for r in evaluable if r["status"] == "needs_review"),
            "insufficient_evidence_count": len(needs_more_evidence),
            "not_evaluated_count": sum(1 for r in results if r["status"] == "not_evaluated"),
            "extraction_suspect_count": len(extraction_suspects),
            "evaluated_raw_score": round(sum(r.get("score") or 0 for r in evaluable), 2),
            "evaluated_possible_score": round(sum(r.get("max_score") or 0 for r in evaluable), 2),
            "note": "missing_variables는 문서 내 단서가 있는데 특정 값만 못 잡은 경우에만 표시합니다.",
        }

    def _recommend(self, results: list[dict[str, Any]], score_gap: float) -> list[dict[str, Any]]:
        candidates = [r for r in results if r["status"] in {"fail", "needs_review", "insufficient_evidence"}]
        candidates.sort(key=lambda r: r.get("max_score") or 0, reverse=True)
        top = candidates[:5]
        return [
            {
                "name": "문서 보완 및 검토 우선 조합",
                "expected_additional_score_range": [0, round(sum((r.get("max_score") or 0) for r in top), 2)],
                "items": [
                    {
                        "criterion_id": r["criterion_id"],
                        "name": r["name"],
                        "current_status": r["status"],
                        "max_score": r.get("max_score"),
                        "reason": r.get("reason"),
                        "required_variables": r.get("required_variables", []),
                    }
                    for r in top
                ],
                "note": f"목표 점수까지 {score_gap:.2f}점이 부족합니다. 다만 현재 실험에서는 점수보다 근거 커버리지와 검토 가능 범위를 우선합니다.",
            }
        ]

    def _missing_documents(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """추가 문서는 항목별로 늘어놓지 않고 문서 종류별로 묶어 제안한다."""
        candidates = [r for r in results if r["status"] == "insufficient_evidence"]
        candidates.sort(key=lambda r: r.get("max_score") or 0, reverse=True)
        grouped: dict[str, dict[str, Any]] = {}

        for r in candidates:
            suggested = self._suggest_document_for_criterion(r)
            group = grouped.setdefault(
                suggested,
                {
                    "suggested_document": suggested,
                    "related_criteria": [],
                    "required_variables": [],
                    "max_score_sum": 0.0,
                },
            )
            group["related_criteria"].append(
                {
                    "criterion_id": r["criterion_id"],
                    "criterion_name": r["name"],
                    "max_score": r.get("max_score", 0),
                }
            )
            group["required_variables"].extend(r.get("required_variables", []))
            group["max_score_sum"] += r.get("max_score") or 0

        docs = list(grouped.values())
        docs.sort(key=lambda item: item.get("max_score_sum") or 0, reverse=True)
        for doc in docs:
            doc["required_variables"] = sorted(set(doc["required_variables"]))[:10]
            doc["max_score_sum"] = round(doc["max_score_sum"], 2)
        return docs[:5]

    def _suggest_document_for_criterion(self, result: dict[str, Any]) -> str:
        """항목명/번호 기반의 간단한 추가 문서 제안."""
        cid = result.get("criterion_id", "")
        name = result.get("name") or ""
        if cid.startswith("R-3") or "자재" in name or "제품" in name:
            return "자재 인증서, 환경성선언/저탄소/환경표지 인증서, 시험성적서"
        if cid.startswith("R-6") or "생태" in name or "녹지" in name:
            return "조경계획표, 자연지반녹지율 산출서, 생태면적률 산출서"
        if cid.startswith("R-4") or "빗물" in name or "물" in name:
            return "빗물관리계획서, 물순환 산출서, 우수/중수 설비 용량 산출서"
        if cid.startswith("R-2") or "에너지" in name:
            return "에너지절약계획 설계검토서, 신재생에너지 설치비율 산출서, 설비 사양서"
        if cid.startswith("R-7") or "실내" in name:
            return "실내공기질 시험성적서, 환기설비 도면, 세대 설비 상세도"
        return "해당 항목 산출서, 관련 도면 또는 인증 증빙"
