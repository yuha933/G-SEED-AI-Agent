from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config
from .extractors import EXTRACTORS
from .gseed_checker import GSeedChecker
from .llm_client import LLMClient
from .pdf_reader import read_pdf
from .report_generator import save_reports


class GSeedAgent:
    """PDF 기반 G-SEED 사전검토 흐름을 관리한다."""

    def __init__(
        self,
        use_ocr: bool = False,
        target_score: float = config.DEFAULT_TARGET_SCORE,
        ocr_max_pages: int | None = 5,
        certification_case: str = config.DEFAULT_CERTIFICATION_CASE,
        target_grade: str | None = config.DEFAULT_TARGET_GRADE,
    ) -> None:
        self.use_ocr = use_ocr
        self.certification_case = certification_case
        self.target_grade = target_grade
        self.target_score = self._resolve_target_score(target_score, certification_case, target_grade)
        self.ocr_max_pages = ocr_max_pages
        self.llm = LLMClient()
        self.checker = GSeedChecker(config.GSEED_DIR)

    def run(self, documents: list[dict[str, Any]], run_name: str | None = None) -> dict[str, Any]:
        """문서 목록을 분석하고 실행 결과를 저장한다."""
        run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = config.RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        document_results = []
        all_results = []

        for doc in documents:
            parsed = read_pdf(
                doc["path"],
                doc.get("document_id"),
                use_ocr=self.use_ocr,
                ocr_max_pages=self.ocr_max_pages,
            ).to_dict()

            # 레지스트리에 문서 유형이 명시된 경우 해당 값을 우선 사용한다.
            parsed["document_type"] = doc.get("document_type") or parsed["document_type"]

            extractor = EXTRACTORS.get(parsed["document_type"])
            if extractor is None:
                extracted = {"facts": [], "candidate_criteria": []}
            else:
                extracted = extractor(parsed)

            result = self.checker.evaluate(
                facts=extracted["facts"],
                candidate_criteria=extracted["candidate_criteria"],
                target_score=self.target_score,
            )
            result["document"] = {
                "document_id": parsed["document_id"],
                "filename": parsed["filename"],
                "document_type": parsed["document_type"],
                "page_count": parsed["page_count"],
                "text_page_count": parsed["text_page_count"],
                "image_only_page_count": parsed["image_only_page_count"],
                "merge_allowed": doc.get("merge_allowed", False),
            }
            result["facts"] = extracted["facts"]

            doc_dir = run_dir / parsed["document_id"]
            doc_dir.mkdir(parents=True, exist_ok=True)
            (doc_dir / "parsed_document.json").write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (doc_dir / "extracted_facts.json").write_text(
                json.dumps(extracted["facts"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            save_reports(result, doc_dir, self.llm)

            document_results.append(result)
            all_results.extend(result["criteria_results"])

        final = self._aggregate(document_results, all_results)
        save_reports(final, run_dir, self.llm)
        return final

    def _aggregate(self, document_results: list[dict[str, Any]], criteria_results: list[dict[str, Any]]) -> dict[str, Any]:
        """문서별 독립 결과를 참고용으로 합산한다."""
        raw_estimated = sum(r.get("score") or 0 for r in criteria_results if r["status"] in {"pass", "needs_review"})
        normalized_estimated = self.checker._normalize_score(raw_estimated)
        evaluated_max_score = sum(r.get("max_score") or 0 for r in criteria_results)
        gap = max(0.0, self.target_score - normalized_estimated)
        counts = self.checker._status_counts(criteria_results)
        coverage_summary = self.checker._coverage_summary(criteria_results)
        grade_info = self._grade_info(normalized_estimated)

        return {
            "summary": {
                "estimated_score": round(normalized_estimated, 2),
                "score_basis": "100점 환산",
                "raw_estimated_score": round(raw_estimated, 2),
                "total_reference_score": round(self.checker.total_reference_score, 2),
                "evaluated_max_score": round(evaluated_max_score, 2),
                "target_score": self.target_score,
                "target_grade": self.target_grade,
                "score_gap": round(gap, 2),
                "certification_case": self.certification_case,
                "certification_case_label": grade_info["case_label"],
                "estimated_certification_grade": grade_info["current_grade"],
                "next_certification_grade": grade_info["next_grade"],
                "next_grade_threshold": grade_info["next_threshold"],
                "next_grade_gap": grade_info["next_gap"],
                "grade_thresholds": grade_info["thresholds"],
                "evaluation_mode": "document_independent",
                "confidence_note": "서로 다른 문서 기반 실험이므로 합산 점수는 참고용입니다.",
            },
            "status_counts": counts,
            "coverage_summary": coverage_summary,
            "criteria_results": criteria_results,
            "document_results": document_results,
            "recommendation_sets": self.checker._recommend(criteria_results, gap),
            "missing_documents": self.checker._missing_documents(criteria_results),
        }

    def _resolve_target_score(self, target_score: float, certification_case: str, target_grade: str | None) -> float:
        """목표 등급이 지정되면 해당 등급 커트라인을 목표점수로 사용한다."""
        if not target_grade:
            return target_score
        case = config.CERTIFICATION_GRADE_THRESHOLDS.get(certification_case)
        if not case:
            return target_score
        return float(case["grades"].get(target_grade, target_score))

    def _grade_info(self, score: float) -> dict[str, Any]:
        """인증등급별 점수기준에 따라 현재/다음 등급 정보를 계산한다."""
        case = config.CERTIFICATION_GRADE_THRESHOLDS.get(self.certification_case)
        if not case:
            return {
                "case_label": self.certification_case,
                "current_grade": "알 수 없음",
                "next_grade": None,
                "next_threshold": None,
                "next_gap": None,
                "thresholds": {},
            }

        thresholds = case["grades"]
        current_grade = "등급 없음"
        for grade, threshold in sorted(thresholds.items(), key=lambda item: item[1], reverse=True):
            if score >= threshold:
                current_grade = grade
                break

        next_grade = None
        next_threshold = None
        for grade, threshold in sorted(thresholds.items(), key=lambda item: item[1]):
            if score < threshold:
                next_grade = grade
                next_threshold = threshold
                break

        return {
            "case_label": case["label"],
            "current_grade": current_grade,
            "next_grade": next_grade,
            "next_threshold": next_threshold,
            "next_gap": None if next_threshold is None else round(max(0.0, next_threshold - score), 2),
            "thresholds": thresholds,
        }


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    """문서 레지스트리를 읽고 상대 경로를 project 폴더 기준으로 해석한다."""
    registry_path = Path(path)
    data = json.loads(registry_path.read_text(encoding="utf-8"))

    documents = data.get("documents", [])
    for doc in documents:
        doc_path = Path(doc["path"])
        if doc_path.is_absolute():
            continue

        project_relative = config.PROJECT_DIR / doc_path
        registry_relative = registry_path.parent / doc_path

        if project_relative.exists():
            doc["path"] = str(project_relative)
        elif registry_relative.exists():
            doc["path"] = str(registry_relative)
        else:
            doc["path"] = str(project_relative)

    return documents
