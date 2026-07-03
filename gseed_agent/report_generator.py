from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm_client import LLMClient


def save_reports(result: dict[str, Any], run_dir: str | Path, llm: LLMClient | None = None) -> dict[str, str]:
    """JSON과 Markdown 보고서를 저장한다."""
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "gseed_precheck_result.json"
    md_path = out_dir / "summary_report.md"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(result, llm), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}


def to_markdown(result: dict[str, Any], llm: LLMClient | None = None) -> str:
    """결과를 Markdown 보고서로 변환한다."""
    summary = result.get("summary", {})
    counts = result.get("status_counts", {})
    coverage = result.get("coverage_summary", {})
    llm_summary = (llm or LLMClient()).write_summary(result)

    lines = [
        "# G-SEED 사전검토 결과",
        "",
        llm_summary,
        "",
        "## 점수 및 등급 요약",
        "",
        f"- 인증 기준 유형: {summary.get('certification_case_label', '-')}",
        f"- 현재 예상 점수: {summary.get('estimated_score', 0):.2f}점 ({summary.get('score_basis', '점수')})",
        f"- 현재 예상 등급: {summary.get('estimated_certification_grade', '-')}",
        f"- 목표 등급: {summary.get('target_grade', '-')}",
        f"- 목표 점수: {summary.get('target_score', 0):.2f}점",
        f"- 목표까지 부족 점수: {summary.get('score_gap', 0):.2f}점",
    ]

    if summary.get("next_certification_grade"):
        lines.append(
            f"- 다음 등급: {summary.get('next_certification_grade')} "
            f"({summary.get('next_grade_threshold'):.2f}점 이상), "
            f"부족 점수 {summary.get('next_grade_gap', 0):.2f}점"
        )

    lines += [
        f"- 원점수 기준 확인 가능 점수: {summary.get('raw_estimated_score', 0):.2f} / {summary.get('total_reference_score', 0):.2f}점",
        f"- 현재 평가 대상 항목 최대 점수: {summary.get('evaluated_max_score', 0):.2f}점",
        "",
        "## 문서 커버리지 요약",
        "",
        f"- 평가 완료 항목: {coverage.get('evaluated_count', 0)}개",
        f"- 사람 검토 필요 항목: {coverage.get('needs_review_count', 0)}개",
        f"- 근거 부족 항목: {coverage.get('insufficient_evidence_count', 0)}개",
        f"- 현재 문서 범위에서 판단 제외: {coverage.get('not_evaluated_count', 0)}개",
        f"- 추출 실패 의심 항목: {coverage.get('extraction_suspect_count', 0)}개",
        f"- 평가 완료 항목 원점수: {coverage.get('evaluated_raw_score', 0):.2f} / {coverage.get('evaluated_possible_score', 0):.2f}점",
        f"- 메모: {coverage.get('note', '-')}",
        "",
        "## 인증등급별 점수기준",
        "",
    ]

    thresholds = summary.get("grade_thresholds", {})
    for grade in ["최우수", "우수", "우량", "일반"]:
        if grade in thresholds:
            lines.append(f"- {grade}: {thresholds[grade]:.0f}점 이상")

    lines += [
        "",
        "## 상태 요약",
        "",
        f"- pass: {counts.get('pass', 0)}",
        f"- fail: {counts.get('fail', 0)}",
        f"- needs_review: {counts.get('needs_review', 0)}",
        f"- insufficient_evidence: {counts.get('insufficient_evidence', 0)}",
        f"- not_evaluated: {counts.get('not_evaluated', 0)}",
        f"- unknown: {counts.get('unknown', 0)}",
        "",
        "## 항목별 결과",
        "",
        "| 항목 | 상태 | 등급 | 점수 | 사유 |",
        "|---|---|---:|---:|---|",
    ]

    for row in result.get("criteria_results", []):
        score = "" if row.get("score") is None else f"{row['score']:.2f}"
        lines.append(
            f"| {row.get('criterion_id')} {row.get('name')} | "
            f"{row.get('status')} | {row.get('grade') or ''} | {score} | {row.get('reason', '')} |"
        )

    lines += ["", "## 추천 조합", ""]
    for rec in result.get("recommendation_sets", []):
        lines.append(f"### {rec.get('name')}")
        lines.append("")
        lines.append(rec.get("note", ""))
        for item in rec.get("items", []):
            lines.append(
                f"- {item.get('criterion_id')} {item.get('name')} "
                f"({item.get('current_status')}, 최대 {item.get('max_score')}점)"
            )
        lines.append("")

    lines += ["## 추가 필요 문서", ""]
    for doc in result.get("missing_documents", []):
        related = doc.get("related_criteria", [])
        related_text = ", ".join(f"{item.get('criterion_id')}" for item in related[:6])
        required = ", ".join(doc.get("required_variables", [])[:6])
        criteria_suffix = f" / 관련 항목: {related_text}" if related_text else ""
        suffix = f" / 필요 변수: {required}" if required else ""
        lines.append(f"- {doc.get('suggested_document')}{criteria_suffix}{suffix}")

    return "\n".join(lines)
