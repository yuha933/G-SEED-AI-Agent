from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .agent import GSeedAgent
from .gseed_kb import load_or_build_gseed_kb
from .llm_client import LLMClient
from .rag import GSeedGraphRetriever
from .workflow import run_gseed_workflow


app = FastAPI(
    title="G-SEED Consulting AI Agent API",
    description="PDF 문서 분석, G-SEED 사전검토, 결과 조회, 챗봇 응답 API",
    version="0.1.0",
)

# Streamlit/브라우저/외부 프론트엔드에서 호출할 수 있게 CORS를 열어둔다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """챗봇 질의 요청."""

    question: str = Field(..., min_length=1)
    run_id: str | None = None
    result: dict[str, Any] | None = None


class ReviewOverrideRequest(BaseModel):
    """수동 검토값 저장 요청.

    현재는 API 사용자가 override를 보존할 수 있게 파일만 저장한다.
    실제 재계산은 이후 LangGraph 단계에서 workflow로 묶는 것이 좋다.
    """

    run_id: str
    overrides: dict[str, str]


@app.get("/health")
def health() -> dict[str, str]:
    """서버 상태 확인용 엔드포인트."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_documents(
    files: list[UploadFile] = File(...),
    document_types: list[str] | None = Form(None),
    use_ocr: bool = Form(True),
    ocr_max_pages: int = Form(25),
    target_grade: str = Form(config.DEFAULT_TARGET_GRADE),
    certification_case: str = Form(config.DEFAULT_CERTIFICATION_CASE),
    use_langgraph: bool = Form(True),
) -> dict[str, Any]:
    """업로드된 PDF들을 분석하고 G-SEED 사전검토 결과를 반환한다."""
    if not files:
        raise HTTPException(status_code=400, detail="분석할 PDF 파일이 없습니다.")

    document_types = document_types or []
    run_id = f"api_{int(time.time())}"
    upload_dir = config.RUNS_DIR / "api_uploads" / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    for idx, upload in enumerate(files):
        if not upload.filename:
            raise HTTPException(status_code=400, detail=f"{idx + 1}번째 파일 이름이 없습니다.")

        filename = _safe_filename(upload.filename)
        if Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail=f"PDF만 업로드할 수 있습니다: {upload.filename}")

        saved_path = upload_dir / f"{idx + 1:02d}_{filename}"
        content = await upload.read()
        saved_path.write_bytes(content)

        documents.append(
            {
                "document_id": f"doc_{idx + 1}",
                "path": str(saved_path),
                "document_type": document_types[idx] if idx < len(document_types) else "unknown",
                "merge_allowed": False,
            }
        )

    if use_langgraph:
        try:
            result = run_gseed_workflow(
                documents=documents,
                use_ocr=use_ocr,
                ocr_max_pages=ocr_max_pages,
                certification_case=certification_case,
                target_grade=target_grade,
                run_name=run_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    else:
        agent = GSeedAgent(
            use_ocr=use_ocr,
            ocr_max_pages=ocr_max_pages,
            certification_case=certification_case,
            target_grade=target_grade,
        )
        result = agent.run(documents, run_name=run_id)
        result.setdefault("summary", {})["workflow_engine"] = "direct"

    return {
        "run_id": run_id,
        "workflow_engine": result.get("summary", {}).get("workflow_engine", "unknown"),
        "uploaded_file_count": len(files),
        "result": result,
    }


@app.get("/runs/{run_id}")
def get_run_result(run_id: str) -> dict[str, Any]:
    """저장된 실행 결과 JSON을 조회한다."""
    result_path = _run_dir(run_id) / "gseed_precheck_result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="해당 run_id의 결과를 찾지 못했습니다.")
    return json.loads(result_path.read_text(encoding="utf-8"))


@app.post("/review/overrides")
def save_review_overrides(request: ReviewOverrideRequest) -> dict[str, Any]:
    """수동 검토 선택값을 run 폴더에 저장한다."""
    run_dir = _run_dir(request.run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="해당 run_id를 찾지 못했습니다.")

    path = run_dir / "review_overrides.json"
    path.write_text(json.dumps(request.overrides, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"run_id": request.run_id, "saved": True, "path": str(path)}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """G-SEED KB와 실행 결과를 참고해 챗봇 답변을 반환한다."""
    result = request.result
    if result is None and request.run_id:
        result = get_run_result(request.run_id)

    kb = load_or_build_gseed_kb()
    graph_retriever = GSeedGraphRetriever(kb, result, max_hops=2)
    context = graph_retriever.retrieve(request.question, limit=4)
    fallback = _fallback_chat_answer(context)

    llm = LLMClient()
    if not llm.enabled:
        return {"answer": fallback, "llm_enabled": False, "rag": context}

    response = llm.complete_json(
        system=(
            "너는 녹색건축 인증기준(G-SEED) 인증 항목 컨설팅 AI Agent다. "
            "제공된 graph RAG context만 근거로 답한다. "
            "START 노드와 그래프 경로의 필요 문서, 확인값, 등급 기준, 현재 프로젝트 근거를 우선 사용한다. "
            "질문과 관련성이 낮은 항목은 나열하지 않는다. "
            "변수명 대신 사용자가 문서에서 확인할 수 있는 표현으로 설명한다."
        ),
        user=json.dumps(
            {
                "question": request.question,
                "graph_rag_context": context,
                "instruction": "answer 필드 하나를 가진 JSON으로 간결하게 답해.",
            },
            ensure_ascii=False,
        ),
        fallback={"answer": fallback},
    )
    return {
        "answer": response.get("answer", fallback),
        "llm_enabled": True,
        "rag": context,
    }


def _run_dir(run_id: str) -> Path:
    """run_id가 runs 폴더 밖으로 나가지 않게 검증한다."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id):
        raise HTTPException(status_code=400, detail="run_id 형식이 올바르지 않습니다.")
    return config.RUNS_DIR / run_id


def _safe_filename(filename: str) -> str:
    """업로드 파일명을 저장 가능한 형태로 정리한다."""
    name = Path(filename).name
    name = re.sub(r"[^\w가-힣.\-() ]+", "_", name)
    return name or "document.pdf"


def _fallback_chat_answer(context: dict[str, Any]) -> str:
    """LLM이 없을 때 사용할 기본 답변."""
    start_nodes = context.get("start_nodes") or []
    if not start_nodes:
        return "관련 G-SEED 항목을 찾지 못했습니다. 항목 번호나 키워드를 조금 더 구체적으로 입력해 주세요."

    primary = start_nodes[0]
    lines = [f"가장 관련 있는 기준은 {primary.get('label')}입니다."]
    if context.get("context_text"):
        lines.append(context["context_text"])
    return "\n".join(lines)
