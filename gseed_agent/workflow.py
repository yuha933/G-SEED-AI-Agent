from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from . import config
from .agent import GSeedAgent
from .extractors import EXTRACTORS
from .pdf_reader import read_pdf
from .report_generator import save_reports


class GSeedWorkflowState(TypedDict, total=False):
    """LangGraph가 단계별로 주고받는 상태."""

    agent: GSeedAgent
    documents: list[dict[str, Any]]
    run_id: str
    run_dir: Path
    parsed_documents: list[dict[str, Any]]
    extracted_documents: list[dict[str, Any]]
    document_results: list[dict[str, Any]]
    criteria_results: list[dict[str, Any]]
    final_result: dict[str, Any]


def run_gseed_workflow(
    *,
    documents: list[dict[str, Any]],
    use_ocr: bool = False,
    ocr_max_pages: int | None = 25,
    certification_case: str = config.DEFAULT_CERTIFICATION_CASE,
    target_grade: str | None = config.DEFAULT_TARGET_GRADE,
    run_name: str | None = None,
) -> dict[str, Any]:
    """LangGraph 기반으로 G-SEED 분석 workflow를 실행한다."""
    agent = GSeedAgent(
        use_ocr=use_ocr,
        ocr_max_pages=ocr_max_pages,
        certification_case=certification_case,
        target_grade=target_grade,
    )
    run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_state: GSeedWorkflowState = {
        "agent": agent,
        "documents": documents,
        "run_id": run_id,
        "parsed_documents": [],
        "extracted_documents": [],
        "document_results": [],
        "criteria_results": [],
    }

    graph = build_gseed_workflow()
    final_state = graph.invoke(initial_state)
    result = final_state["final_result"]
    result.setdefault("summary", {})["workflow_engine"] = "langgraph"
    return result


def build_gseed_workflow() -> Any:
    """G-SEED 분석 LangGraph를 만든다.

    langgraph가 설치되어 있지 않으면 실행 시점에 명확한 안내 오류를 낸다.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - 설치 환경 의존
        raise RuntimeError(
            "LangGraph가 설치되어 있지 않습니다. "
            "`pip install -r requirements.txt` 실행 후 다시 시도해 주세요."
        ) from exc

    workflow = StateGraph(GSeedWorkflowState)
    workflow.add_node("prepare_run", _prepare_run)
    workflow.add_node("parse_documents", _parse_documents)
    workflow.add_node("extract_facts", _extract_facts_node)
    workflow.add_node("evaluate_documents", _evaluate_documents)
    workflow.add_node("aggregate_results", _aggregate_results)

    workflow.set_entry_point("prepare_run")
    workflow.add_edge("prepare_run", "parse_documents")
    workflow.add_edge("parse_documents", "extract_facts")
    workflow.add_edge("extract_facts", "evaluate_documents")
    workflow.add_edge("evaluate_documents", "aggregate_results")
    workflow.add_edge("aggregate_results", END)
    return workflow.compile()


def _prepare_run(state: GSeedWorkflowState) -> GSeedWorkflowState:
    """실행 폴더와 상태 기본값을 준비한다."""
    run_id = state["run_id"]
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state["run_dir"] = run_dir
    state.setdefault("parsed_documents", [])
    state.setdefault("extracted_documents", [])
    state.setdefault("document_results", [])
    state.setdefault("criteria_results", [])
    return state


def _parse_documents(state: GSeedWorkflowState) -> GSeedWorkflowState:
    """모든 PDF를 페이지 단위 구조화 JSON으로 변환한다."""
    agent = state["agent"]
    parsed_documents: list[dict[str, Any]] = []

    for document in state["documents"]:
        parsed = _parse_document(agent, document)
        parsed["_source_document"] = document
        parsed_documents.append(parsed)

    state["parsed_documents"] = parsed_documents
    return state


def _extract_facts_node(state: GSeedWorkflowState) -> GSeedWorkflowState:
    """구조화된 문서에서 설계값 후보를 추출한다."""
    extracted_documents: list[dict[str, Any]] = []
    for parsed in state["parsed_documents"]:
        extracted = _extract_facts(parsed)
        extracted_documents.append(
            {
                "parsed": parsed,
                "extracted": extracted,
                "source_document": parsed.get("_source_document", {}),
            }
        )
    state["extracted_documents"] = extracted_documents
    return state


def _evaluate_documents(state: GSeedWorkflowState) -> GSeedWorkflowState:
    """추출된 설계값과 G-SEED 기준을 비교하고 문서별 결과를 저장한다."""
    agent = state["agent"]
    run_dir = state["run_dir"]
    document_results: list[dict[str, Any]] = []
    criteria_results: list[dict[str, Any]] = []

    for item in state["extracted_documents"]:
        parsed = item["parsed"]
        extracted = item["extracted"]
        source_document = item["source_document"]
        result = _evaluate_document(agent, parsed, extracted, source_document)
        _save_document_outputs(agent, run_dir, parsed, extracted, result)

        document_results.append(result)
        criteria_results.extend(result["criteria_results"])

    state["document_results"] = document_results
    state["criteria_results"] = criteria_results
    return state


def _aggregate_results(state: GSeedWorkflowState) -> GSeedWorkflowState:
    """문서별 결과를 최종 결과로 합산하고 저장한다."""
    agent = state["agent"]
    run_dir = state["run_dir"]
    final = agent._aggregate(state["document_results"], state["criteria_results"])
    final.setdefault("summary", {})["workflow_engine"] = "langgraph"
    final.setdefault("summary", {})["workflow_nodes"] = [
        "prepare_run",
        "parse_documents",
        "extract_facts",
        "evaluate_documents",
        "aggregate_results",
    ]
    save_reports(final, run_dir, agent.llm)
    state["final_result"] = final
    return state


def _parse_document(agent: GSeedAgent, document: dict[str, Any]) -> dict[str, Any]:
    """PDF를 페이지 단위 구조화 JSON으로 변환한다."""
    parsed = read_pdf(
        document["path"],
        document.get("document_id"),
        use_ocr=agent.use_ocr,
        ocr_max_pages=agent.ocr_max_pages,
    ).to_dict()
    parsed["document_type"] = document.get("document_type") or parsed["document_type"]
    return parsed


def _extract_facts(parsed: dict[str, Any]) -> dict[str, Any]:
    """문서 유형별 extractor를 실행한다."""
    extractor = EXTRACTORS.get(parsed["document_type"])
    if extractor is None:
        return {"facts": [], "candidate_criteria": []}
    return extractor(parsed)


def _evaluate_document(
    agent: GSeedAgent,
    parsed: dict[str, Any],
    extracted: dict[str, Any],
    source_document: dict[str, Any],
) -> dict[str, Any]:
    """추출된 설계값과 G-SEED 기준을 비교한다."""
    result = agent.checker.evaluate(
        facts=extracted["facts"],
        candidate_criteria=extracted["candidate_criteria"],
        target_score=agent.target_score,
    )
    result["document"] = {
        "document_id": parsed["document_id"],
        "filename": parsed["filename"],
        "document_type": parsed["document_type"],
        "page_count": parsed["page_count"],
        "text_page_count": parsed["text_page_count"],
        "image_only_page_count": parsed["image_only_page_count"],
        "merge_allowed": source_document.get("merge_allowed", False),
    }
    result["facts"] = extracted["facts"]
    return result


def _save_document_outputs(
    agent: GSeedAgent,
    run_dir: Path,
    parsed: dict[str, Any],
    extracted: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """문서별 구조화 결과, 설계값 후보, 평가 결과를 저장한다."""
    doc_dir = run_dir / parsed["document_id"]
    doc_dir.mkdir(parents=True, exist_ok=True)
    parsed_for_save = {key: value for key, value in parsed.items() if not key.startswith("_")}
    (doc_dir / "parsed_document.json").write_text(
        json.dumps(parsed_for_save, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (doc_dir / "extracted_facts.json").write_text(
        json.dumps(extracted["facts"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_reports(result, doc_dir, agent.llm)
