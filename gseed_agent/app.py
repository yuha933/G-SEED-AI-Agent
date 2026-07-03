from __future__ import annotations

import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from . import config
    from .agent import GSeedAgent
    from .gseed_kb import load_or_build_gseed_kb
    from .llm_client import LLMClient
    from .rag import GSeedGraphRetriever
except ImportError:
    # streamlit run으로 파일을 직접 실행할 때 import 경로를 보정한다.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gseed_agent import config
    from gseed_agent.agent import GSeedAgent
    from gseed_agent.gseed_kb import load_or_build_gseed_kb
    from gseed_agent.llm_client import LLMClient
    from gseed_agent.rag import GSeedGraphRetriever


UPLOAD_SLOTS = [
    {
        "key": "building_drawing",
        "label": "건축도면 통합본",
        "document_type": "drawing",
        "help": [
            "대지, 면적, 층수, 지하개발, 주차, 재활용보관시설",
            "세대 평면, 피난, 복도, 계단, 커뮤니티시설",
            "일부 주택성능 항목",
        ],
    },
    {
        "key": "facility_drawing",
        "label": "설비도면 통합본",
        "document_type": "drawing",
        "help": [
            "환기, 자동온도조절, 급수압, 감압밸브",
            "에너지 모니터링, 물 사용량 모니터링",
            "홈네트워크, 방범, 감지경보, 제연설비",
        ],
    },
    {
        "key": "energy_plan",
        "label": "에너지절약계획서/설계검토서",
        "document_type": "energy",
        "help": [
            "에너지 성능",
            "신재생에너지",
            "저탄소 에너지원",
            "냉매/단열재/보일러 관련 항목",
        ],
    },
    {
        "key": "landscape_plan",
        "label": "조경계획표/생태면적률 산출서",
        "document_type": "landscape",
        "help": [
            "자연지반녹지율",
            "생태면적률",
            "비오톱",
            "녹지축",
            "표토재활용 일부",
        ],
    },
    {
        "key": "material_certificates",
        "label": "마감자재목록표 + 자재 인증서",
        "document_type": "material",
        "help": [
            "EPD",
            "저탄소 자재",
            "자원순환 자재",
            "유해물질 저감 자재",
            "실내공기 오염물질 저방출 제품",
        ],
    },
    {
        "key": "construction_plan",
        "label": "시공계획서/환경관리계획서",
        "document_type": "construction",
        "help": [
            "건설현장 환경관리",
            "녹색 건설현장 환경관리 수행",
        ],
    },
    {
        "key": "maintenance_manual",
        "label": "운영·유지관리/사용자 매뉴얼",
        "document_type": "manual",
        "help": [
            "운영·유지관리 문서",
            "사용자 매뉴얼",
            "녹색건축 인증 정보 제공",
        ],
    },
    {
        "key": "performance_reports",
        "label": "성능시험성적서 묶음",
        "document_type": "performance",
        "help": [
            "충격음",
            "차음",
            "교통소음",
            "급배수소음",
            "내구성/내화성능 일부",
        ],
    },
    {
        "key": "other_documents",
        "label": "그 외 문서",
        "document_type": "other",
        "help": [
            "교통영향평가·대중교통 접근성 자료",
            "폐기물 처리계획서·재활용 계획서",
            "빗물이용·중수도·물 사용량 산출서",
            "실내공기질·소음·음향 관련 시험자료",
            "인증 신청서·자체평가서·기타 근거자료",
        ],
    },
]


DEFAULT_STREAMLIT_OCR_MAX_PAGES = 25
STEPS = ["문서 업로드", "추출 결과 검토", "점수 계산 결과", "개선안 추천", "AI 챗봇"]


st.set_page_config(page_title="G-SEED 컨설팅 AI Agent", layout="wide")


def main() -> None:
    _init_state()
    _inject_css()
    _sidebar()

    page = st.session_state.page
    if page == "문서 업로드":
        _page_upload()
    elif page == "추출 결과 검토":
        _page_review()
    elif page == "점수 계산 결과":
        _page_score()
    elif page == "개선안 추천":
        _page_recommend()
    elif page == "AI 챗봇":
        _page_chat()


def _init_state() -> None:
    defaults = {
        "page": "문서 업로드",
        "analysis_result": None,
        "review_overrides": {},
        "review_target": None,
        "uploaded_paths": [],
        "uploaded_document_summary": [],
        "chat_history": [],
        "kb": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sidebar() -> None:
    with st.sidebar:
        st.title("녹색건축 인증기준 (G-SEED)")
        st.markdown("### 인증 항목 컨설팅 AI Agent")
        st.divider()

        st.markdown("#### 진행 흐름")
        current_index = STEPS.index(st.session_state.page)
        for idx, step in enumerate(STEPS):
            state_class = "active" if idx == current_index else "done" if idx < current_index else "wait"
            mark = "●" if idx == current_index else "✓" if idx < current_index else str(idx + 1)
            cols = st.columns([0.16, 0.84], gap="small")
            cols[0].markdown(
                f"""
                <div class="timeline-node {state_class}">
                    <div class="timeline-dot">{mark}</div>
                    <div class="timeline-line {'hidden' if idx == len(STEPS) - 1 else state_class}"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            button_type = "primary" if idx <= current_index else "secondary"
            if cols[1].button(step, key=f"nav_step_{idx}", use_container_width=True, type=button_type):
                st.session_state.page = step
                st.rerun()

        st.divider()
        st.caption("계산은 규칙 엔진이 수행하고, 설명·추천·질의응답은 LLM/RAG가 보조합니다.")


def _inject_css() -> None:
    """업로드 설명과 표의 밀도를 조금 낮춰 화면을 더 작게 쓴다."""
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] details summary p {
            font-size: 1.02rem !important;
            font-weight: 700 !important;
        }
        .upload-help {
            font-size: 0.68rem;
            line-height: 1.05;
            margin-top: -0.25rem;
            margin-bottom: 0.35rem;
            color: #4b5563;
            padding-left: 1.05rem;
        }
        .upload-help li {
            margin-bottom: 0.01rem;
        }
        .timeline-node {
            width: 100%;
            min-height: 2.65rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 0.16rem;
        }
        .timeline-dot {
            width: 1.24rem;
            height: 1.24rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.68rem;
            font-weight: 800;
            background: #e5e7eb;
            color: #6b7280;
            border: 2px solid #e5e7eb;
        }
        .timeline-line {
            width: 3px;
            flex: 1;
            min-height: 1.18rem;
            background: #e5e7eb;
            margin-top: 0.12rem;
            border-radius: 999px;
        }
        .timeline-node.done .timeline-dot,
        .timeline-node.active .timeline-dot {
            background: #16a34a;
            border-color: #16a34a;
            color: white;
        }
        .timeline-line.done,
        .timeline-line.active {
            background: #16a34a;
        }
        .timeline-line.hidden {
            visibility: hidden;
        }
        div[data-testid="stMarkdownContainer"] table {
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_upload() -> None:
    st.header("1. 문서 업로드")
    st.caption("9개 문서 묶음 중 가지고 있는 자료를 업로드하세요. 같은 유형에 여러 PDF를 넣을 수 있습니다.")

    target_grade = st.selectbox("목표 등급", ["일반", "우량", "우수", "최우수"], index=0)
    certification_case = st.selectbox(
        "인증 기준 유형",
        list(config.CERTIFICATION_GRADE_THRESHOLDS.keys()),
        format_func=lambda key: config.CERTIFICATION_GRADE_THRESHOLDS[key]["label"],
    )
    use_ocr = st.checkbox("이미지형 PDF에 OCR 적용", value=True)

    st.divider()
    uploaded_by_slot: dict[str, list[Any]] = {}

    for slot in UPLOAD_SLOTS:
        with st.expander(slot["label"], expanded=True):
            help_html = "".join(
                (
                    "<li style='font-size:11px; line-height:1.28; "
                    "margin:0 0 2px 0; color:#4b5563;'>"
                    f"{item}</li>"
                )
                for item in slot["help"]
            )
            st.markdown(
                (
                    "<ul class='upload-help' "
                    "style='font-size:11px; line-height:1.28; "
                    "margin:-6px 0 6px 0; padding-left:17px; color:#4b5563;'>"
                    f"{help_html}</ul>"
                ),
                unsafe_allow_html=True,
            )
            uploaded_by_slot[slot["key"]] = st.file_uploader(
                f"{slot['label']} 업로드",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"upload_{slot['key']}",
                label_visibility="collapsed",
            )

    if st.button("문서 분석 시작", type="primary", use_container_width=True):
        documents = _save_uploaded_documents(uploaded_by_slot)
        if not documents:
            st.warning("분석할 PDF를 하나 이상 업로드해줘.")
            return

        with st.spinner("PDF 읽기/OCR/설계값 추출/G-SEED 비교를 수행하는 중입니다..."):
            agent = GSeedAgent(
                use_ocr=use_ocr,
                ocr_max_pages=DEFAULT_STREAMLIT_OCR_MAX_PAGES,
                certification_case=certification_case,
                target_grade=target_grade,
            )
            run_name = f"streamlit_{int(time.time())}"
            result = agent.run(documents, run_name=run_name)

        st.session_state.analysis_result = result
        st.session_state.review_overrides = {}
        st.session_state.review_target = None
        st.session_state.page = "추출 결과 검토"
        st.success("분석이 완료됐어. 추출 결과 검토 화면으로 이동해줘.")
        st.rerun()


def _page_review() -> None:
    st.header("2. 추출 결과 검토")
    result = _require_result()
    if result is None:
        return

    reviewed = _apply_manual_overrides(result)
    rows = _review_rows(reviewed, include_all=True)

    st.caption("자동 추출 결과를 검토하고, 필요한 항목은 [수정]을 눌러 등급/판정 상태를 직접 지정하세요.")
    _uploaded_document_notice()
    _coverage_metrics(reviewed)

    st.divider()
    for row in rows:
        cols = st.columns([2.8, 1.0, 1.0, 1.0, 0.7])
        cols[0].markdown(f"**{row['항목']}**")
        cols[1].write(row["상태"])
        cols[2].write(row["점수"])
        cols[3].write(row["등급"])
        if cols[4].button("수정", key=f"edit_{row['criterion_id']}"):
            st.session_state.review_target = row["criterion_id"]

    if st.session_state.review_target:
        _show_review_editor(st.session_state.review_target, reviewed)

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("점수 계산 결과로 이동", type="primary", use_container_width=True):
        st.session_state.page = "점수 계산 결과"
        st.rerun()
    if c2.button("수동 검토 초기화", use_container_width=True):
        st.session_state.review_overrides = {}
        st.session_state.review_target = None
        st.rerun()


def _page_score() -> None:
    st.header("3. 점수 계산 결과")
    result = _require_result()
    if result is None:
        return
    reviewed = _apply_manual_overrides(result)

    summary = reviewed["summary"]
    _uploaded_document_notice()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재 예상 점수", f"{summary.get('estimated_score', 0):.2f}점")
    c2.metric("현재 등급", summary.get("estimated_certification_grade", "-"))
    c3.metric("목표 등급", summary.get("target_grade", "-"))
    c4.metric("부족 점수", f"{summary.get('score_gap', 0):.2f}점")

    _coverage_metrics(reviewed)

    st.subheader("전체 항목 기준 분류")
    overall = _overall_classification(reviewed)
    _safe_markdown_table(
        [
            {"구분": "전체 G-SEED 항목", "항목 수": overall["total"]},
            {"구분": "평가 충족 항목", "항목 수": len(overall["satisfied"])},
            {"구분": "근거 일부 부족/검토 필요 항목", "항목 수": len(overall["needs_more"])},
            {"구분": "조건 불충족 항목", "항목 수": len(overall["failed"])},
            {"구분": "변수 추출 실패/미확보 항목", "항목 수": len(overall["missing"])},
        ]
    )

    st.subheader("항목별 결과")
    _safe_markdown_table(_result_rows(reviewed))

    st.download_button(
        "JSON 결과 다운로드",
        data=json.dumps(reviewed, ensure_ascii=False, indent=2),
        file_name="gseed_precheck_result_reviewed.json",
        mime="application/json",
    )


def _page_recommend() -> None:
    st.header("4. 개선안 추천")
    result = _require_result()
    if result is None:
        return
    reviewed = _apply_manual_overrides(result)

    summary = reviewed["summary"]
    st.info(
        f"현재 예상 점수는 {summary.get('estimated_score', 0):.2f}점이고, "
        f"다음 등급({summary.get('next_certification_grade')})까지 "
        f"{summary.get('next_grade_gap', 0):.2f}점이 부족합니다."
    )

    st.subheader("보완이 필요한 문서")
    missing_document_names = _missing_document_names(reviewed)
    if missing_document_names:
        st.markdown("\n".join(f"- {name}" for name in missing_document_names))
    else:
        st.caption("현재 결과 기준으로 별도 보완 문서 묶음이 도출되지 않았습니다.")

    st.subheader("추천 조합")
    for rec in _recommendation_cards(reviewed):
        with st.container(border=True):
            st.markdown(f"### {rec['name']}")
            st.caption(rec["note"])
            _safe_markdown_table(rec["rows"], max_cell_chars=220)

    st.subheader("내가 보는 현실적인 다음 액션")
    st.markdown(
        """
        1. 먼저 `needs_review` 항목의 근거 페이지를 사람이 확인해서 확정합니다.
        2. 그다음 고배점 근거 부족 항목인 생태면적률, 실내공기질/자재 인증, 빗물관리 문서를 우선 보완합니다.
        3. 설계 변경이 필요한 항목과 단순 증빙 보완 항목을 분리해 비용 대비 효과를 비교합니다.
        """
    )

    if st.button("챗봇에게 자세히 물어보기", type="primary"):
        st.session_state.page = "AI 챗봇"
        st.rerun()


def _page_chat() -> None:
    st.header("5. AI 챗봇")
    result = _require_result(show_warning=False)
    kb = _kb()
    llm = LLMClient()

    if llm.enabled:
        st.caption("LLM이 활성화되어 있습니다. G-SEED 계층형 KB와 현재 프로젝트 추출 결과를 함께 참고해 답변합니다.")
    else:
        st.warning("OPENAI_API_KEY가 없어 LLM은 비활성화 상태입니다. 현재는 G-SEED KB 검색 기반 fallback 답변만 제공합니다.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("예: 생태면적률 점수 올리려면 뭐가 필요해?")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    answer = _answer_chat(prompt, kb, result, llm)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)


def _save_uploaded_documents(uploaded_by_slot: dict[str, list[Any]]) -> list[dict[str, Any]]:
    upload_root = config.RUNS_DIR / "streamlit_uploads" / str(int(time.time()))
    upload_root.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    uploaded_summary: list[dict[str, str]] = []
    for slot in UPLOAD_SLOTS:
        files = uploaded_by_slot.get(slot["key"]) or []
        for idx, file in enumerate(files, start=1):
            safe_name = f"{slot['key']}_{idx}_{file.name}"
            path = upload_root / safe_name
            path.write_bytes(file.getbuffer())
            documents.append(
                {
                    "document_id": f"{slot['key']}_{idx}",
                    "path": str(path),
                    "document_type": slot["document_type"],
                    "merge_allowed": False,
                    "slot_label": slot["label"],
                }
            )
            uploaded_summary.append(
                {
                    "문서 묶음": slot["label"],
                    "파일명": file.name,
                    "문서 유형": slot["document_type"],
                }
            )
    st.session_state.uploaded_paths = [doc["path"] for doc in documents]
    st.session_state.uploaded_document_summary = uploaded_summary
    return documents


def _uploaded_document_notice() -> None:
    """현재 결과가 업로드 문서 기준임을 UI에서 명확히 보여준다."""
    summary = st.session_state.get("uploaded_document_summary") or []
    if not summary:
        return
    with st.expander("이번 분석에 사용된 업로드 문서", expanded=False):
        _safe_markdown_table(summary)
        st.caption("이 Streamlit 실행 결과는 document_registry.json의 기본 파일이 아니라 위 업로드 파일만 기준으로 계산됩니다.")


def _safe_markdown_table(rows: list[dict[str, Any]] | dict[str, Any] | None, max_cell_chars: int = 160) -> None:
    """pandas/numpy 의존 없이 표를 표시한다.

    Streamlit의 st.dataframe은 내부에서 pandas를 import하므로,
    pandas/numpy 바이너리 충돌 환경에서는 오류가 날 수 있다.
    """
    if rows is None:
        st.caption("표시할 데이터가 없습니다.")
        return
    if isinstance(rows, dict):
        rows = [{"key": key, "value": value} for key, value in rows.items()]
    if not rows:
        st.caption("표시할 데이터가 없습니다.")
        return

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(_escape_md(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [_format_cell(row.get(header), max_cell_chars=max_cell_chars) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    st.markdown("\n".join(lines))


def _format_cell(value: Any, max_cell_chars: int = 160) -> str:
    if value is None:
        text = ""
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > max_cell_chars:
        text = text[: max_cell_chars - 1] + "…"
    return _escape_md(text)


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _require_result(show_warning: bool = True) -> dict[str, Any] | None:
    result = st.session_state.analysis_result
    if result is None and show_warning:
        st.warning("먼저 문서 업로드 화면에서 분석을 실행해줘.")
    return result


def _review_rows(result: dict[str, Any], include_all: bool = False) -> list[dict[str, Any]]:
    rows = []
    criteria_rows = _all_criteria_rows(result) if include_all else result.get("criteria_results", [])
    for row in criteria_rows:
        rows.append(
            {
                "criterion_id": row.get("criterion_id"),
                "항목": f"{row.get('criterion_id')} {row.get('name')}",
                "상태": _display_status(row),
                "점수": "-" if row.get("score") is None else f"{row.get('score'):.2f}점",
                "등급": row.get("grade") or "-",
            }
        )
    return rows


def _result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _all_criteria_rows(result):
        rows.append(
            {
                "항목": f"{row.get('criterion_id')} {row.get('name')}",
                "상태": _display_status(row),
                "등급": row.get("grade"),
                "점수": row.get("score"),
                "최대점수": row.get("max_score"),
                "사유": row.get("reason"),
            }
        )
    return rows


def _all_criteria_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """R-1.1부터 끝까지 전체 67개 항목을 검토표에 표시한다."""
    kb = _kb()
    result_by_id = {row.get("criterion_id"): row for row in result.get("criteria_results", [])}
    rows: list[dict[str, Any]] = []
    for criterion_id in _sorted_criterion_ids(kb.get("criteria_index", {}).keys()):
        if criterion_id in result_by_id:
            rows.append(result_by_id[criterion_id])
        else:
            rows.append(_empty_result_row(criterion_id))
    return rows


def _empty_result_row(criterion_id: str) -> dict[str, Any]:
    """아직 자동 추출 후보가 아닌 항목도 사람이 검토할 수 있게 빈 row를 만든다."""
    criterion = _kb().get("criteria_index", {}).get(criterion_id, {})
    return {
        "criterion_id": criterion_id,
        "name": criterion.get("name"),
        "status": "not_evaluated",
        "grade": None,
        "score": None,
        "max_score": _criterion_max_score_from_kb(criterion),
        "evidence": [],
        "missing_variables": [],
        "required_variables": [v.get("name") for v in criterion.get("input_variables", []) if v.get("name")],
        "reason": "아직 자동 추출 후보로 연결되지 않은 항목입니다. 필요하면 수동 검토로 판정할 수 있습니다.",
    }


def _criterion_max_score_from_kb(criterion: dict[str, Any]) -> float:
    try:
        return float(criterion.get("score") or 0)
    except Exception:
        return 0.0


def _sorted_criterion_ids(ids: Any) -> list[str]:
    """R-1.1, R-1.2 ... R-ID-57 순서로 정렬한다."""
    def key(cid: str) -> tuple[int, float, int]:
        if cid.startswith("R-ID-"):
            return (99, float(cid.replace("R-ID-", "")), 0)
        match = re.search(r"R-(\d+)\.(\d+)", cid)
        if match:
            return (int(match.group(1)), float(match.group(2)), 0)
        return (98, 0, 0)

    return sorted(ids, key=key)


def _manual_options_for_criterion(criterion: dict[str, Any]) -> list[str]:
    """해당 항목 기준표에 실제 존재하는 선택지만 반환한다."""
    options: list[str] = []
    seen: set[str] = set()
    for rule in criterion.get("grade_rules", []):
        grade = rule.get("grade")
        if not grade:
            continue
        label = str(grade)
        if label not in seen:
            options.append(label)
            seen.add(label)

    # 등급표가 없는 점수형/체크형 항목은 수동 충족만 제공한다.
    if not options:
        max_score = _criterion_max_score_from_kb(criterion)
        if max_score:
            options.append(f"충족 ({max_score:g}점)")
        else:
            options.append("충족")

    options.extend(["미충족", "판단불가"])
    return options


def _current_manual_option(row: dict[str, Any], criterion: dict[str, Any]) -> str | None:
    """수정 팝업에서 현재 선택된 판정 버튼을 강조하기 위한 값."""
    criterion_id = row.get("criterion_id")
    override = st.session_state.review_overrides.get(criterion_id)
    if override:
        return override

    grade = row.get("grade")
    if grade:
        for option in _manual_options_for_criterion(criterion):
            if option == grade or option.startswith(str(grade)):
                return option

    status = row.get("status")
    if status == "fail":
        return "미충족"
    if status in {"unknown", "not_evaluated", "insufficient_evidence"}:
        return "판단불가"
    if status in {"pass", "needs_review"}:
        for option in _manual_options_for_criterion(criterion):
            if option.startswith("충족"):
                return option
    return None


def _reference_document_rows(criterion: dict[str, Any]) -> list[dict[str, Any]]:
    """수정 창에서 보여줄 참고자료/제출서류 표를 만든다."""
    rows: list[dict[str, Any]] = []
    required_documents = criterion.get("required_documents", {}) or {}
    for phase, docs in required_documents.items():
        if not docs:
            continue
        rows.append(
            {
                "구분": f"제출서류-{phase}",
                "내용": " / ".join(str(doc) for doc in docs),
            }
        )

    references = criterion.get("references", []) or []
    if references:
        rows.append(
            {
                "구분": "참고자료",
                "내용": " / ".join(str(ref) for ref in references),
            }
        )

    variables = criterion.get("input_variables", []) or []
    if variables:
        rows.append(
            {
                "구분": "확인해야 할 설계값",
                "내용": " / ".join((v.get("label") or v.get("name") or "") for v in variables[:12]),
            }
        )
    return rows


def _missing_document_names(result: dict[str, Any]) -> list[str]:
    """개선안 화면에는 문서 묶음 이름만 간단히 보여준다."""
    names: list[str] = []
    for item in result.get("missing_documents", []):
        name = item.get("suggested_document")
        if name and name not in names:
            names.append(name)
    return names


def _recommendation_cards(result: dict[str, Any]) -> list[dict[str, Any]]:
    """개발자용 추천 데이터를 사용자용 개선안 카드 4개로 바꾼다."""
    rows = [
        row
        for row in _all_criteria_rows(result)
        if row.get("status") in {"fail", "needs_review", "insufficient_evidence", "not_evaluated"}
    ]
    rows.sort(key=lambda row: row.get("max_score") or 0, reverse=True)

    evidence_rows = [row for row in rows if row.get("status") in {"needs_review", "insufficient_evidence"}]
    high_score_rows = rows[:5]
    cost_rows = [
        row
        for row in rows
        if _document_for_row(row) in {
            "마감자재목록표 + 자재 인증서",
            "운영·유지관리/사용자 매뉴얼",
            "성능시험성적서 묶음",
            "시공계획서/환경관리계획서",
        }
    ][:5]
    design_rows = [
        row
        for row in rows
        if _document_for_row(row) in {"건축도면 통합본", "설비도면 통합본", "조경계획표/생태면적률 산출서"}
    ][:5]

    return [
        {
            "name": "문서 보완형",
            "note": "설계 변경보다 산출서·인증서·성적서 보완으로 먼저 확인할 수 있는 항목입니다.",
            "rows": _recommendation_rows(evidence_rows[:5]),
        },
        {
            "name": "고배점 영향형",
            "note": "점수 영향이 큰 항목부터 확인해 부족 점수 회복 가능성을 보는 조합입니다.",
            "rows": _recommendation_rows(high_score_rows),
        },
        {
            "name": "비용 대비 효과형",
            "note": "상대적으로 문서 보완이나 부분 확인으로 판정 가능성이 높은 조합입니다.",
            "rows": _recommendation_rows(cost_rows or evidence_rows[:5]),
        },
        {
            "name": "설계 검토형",
            "note": "공간·조경·설비 계획 자체를 확인하거나 일부 설계 조정이 필요할 수 있는 조합입니다.",
            "rows": _recommendation_rows(design_rows or high_score_rows),
        },
    ]


def _recommendation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return [{"관련 항목": "-", "필요 문서": "추가 확인 필요", "문서에서 볼 내용": "현재 추출 결과만으로는 추천 대상을 특정하기 어렵습니다."}]

    return [
        {
            "관련 항목": f"{row.get('criterion_id')} {row.get('name')}",
            "필요 문서": _document_for_row(row),
            "문서에서 볼 내용": " / ".join(_human_check_items_for_row(row)[:5]),
        }
        for row in rows
    ]


def _document_for_row(row: dict[str, Any]) -> str:
    """항목명/번호를 사용자가 아는 문서 묶음 이름으로 변환한다."""
    cid = row.get("criterion_id", "")
    name = row.get("name") or ""
    if cid.startswith("R-3") or "자재" in name or "제품" in name:
        return "마감자재목록표 + 자재 인증서"
    if cid.startswith("R-6") or "생태" in name or "녹지" in name or "비오톱" in name:
        return "조경계획표/생태면적률 산출서"
    if cid.startswith("R-4") or "빗물" in name or "물" in name:
        return "설비도면 통합본 또는 물순환 산출서"
    if cid.startswith("R-2") or "에너지" in name or "신재생" in name:
        return "에너지절약계획서/설계검토서"
    if cid.startswith("R-7") or "실내" in name or "소음" in name or "환기" in name:
        return "설비도면 통합본 또는 성능시험성적서 묶음"
    if cid.startswith("R-8"):
        return "건축도면 통합본 또는 성능시험성적서 묶음"
    if cid.startswith("R-5"):
        return "운영·유지관리/사용자 매뉴얼"
    return "건축도면 통합본 또는 기타 근거자료"


def _human_check_items_for_row(row: dict[str, Any]) -> list[str]:
    """변수명이 아니라 문서 안에서 사람이 확인할 표현으로 바꾼다."""
    cid = row.get("criterion_id", "")
    name = row.get("name") or ""

    if cid == "R-6.3" or "생태면적률" in name:
        return ["생태면적률 산출값", "피복유형별 면적", "식재유형별 면적", "대지면적", "환산면적 합계"]
    if cid == "R-6.2" or "자연지반" in name:
        return ["자연지반 녹지 면적", "전체 대지면적", "녹지율 산출식"]
    if "비오톱" in name:
        return ["비오톱 조성 면적", "수생·육생 비오톱 종류", "공통 적용항목 충족 여부"]
    if "에너지" in name or "신재생" in name:
        return ["에너지 성능 지표", "신재생에너지 설치비율", "설비 사양", "산출 근거"]
    if "자재" in name or "제품" in name:
        return ["적용 자재 목록", "인증서 종류", "적용 위치", "자재비 또는 적용 수량"]
    if "빗물" in name or "물" in name:
        return ["시설 용량", "연간 물 사용량", "절수 계획", "계통도"]
    if "소음" in name or "차음" in name or "충격음" in name:
        return ["시험성적서 등급", "적용 세대 또는 공간", "측정 기준", "성능값"]
    if "환기" in name:
        return ["환기설비 방식", "세대별 적용 여부", "환기량 또는 성능 등급"]

    labels = []
    for item in row.get("required_variables", [])[:5]:
        text = str(item).replace("_", " ")
        if text and text not in labels:
            labels.append(text)
    return labels or ["해당 항목 산출값", "관련 도면 표기", "제출서류 근거"]


def _parse_manual_grade_option(option: str, criterion_id: str | None = None) -> tuple[str | None, float]:
    """선택지 문자열에서 등급과 가중치를 읽는다."""
    if option.startswith("충족"):
        return "충족", 1.0
    grade_match = re.search(r"(\d+급)", option)
    if not grade_match:
        return None, 0.0
    grade = grade_match.group(1)
    if criterion_id:
        criterion = _kb().get("criteria_index", {}).get(criterion_id, {})
        for rule in criterion.get("grade_rules", []):
            if str(rule.get("grade")) == grade and rule.get("weight") is not None:
                return grade, float(rule.get("weight"))
    return grade, {"1급": 1.0, "2급": 0.8, "3급": 0.6, "4급": 0.4}.get(grade, 1.0)


def _show_review_editor(criterion_id: str, result: dict[str, Any]) -> None:
    """가능하면 Streamlit dialog로 수동 검토창을 띄운다."""
    if hasattr(st, "dialog"):
        _review_editor_dialog(criterion_id, result)
    else:
        _review_editor_content(criterion_id, result)


if hasattr(st, "dialog"):
    @st.dialog("항목 수동 검토", width="large")
    def _review_editor_dialog(criterion_id: str, result: dict[str, Any]) -> None:
        _review_editor_content(criterion_id, result)


def _review_editor_content(criterion_id: str, result: dict[str, Any]) -> None:
    row = next((r for r in _all_criteria_rows(result) if r.get("criterion_id") == criterion_id), None)
    criterion = _kb().get("criteria_index", {}).get(criterion_id)
    if not row or not criterion:
        return

    with st.container(border=True):
        st.subheader(f"{criterion_id} {row.get('name')}")
        st.caption("기준표를 확인하고 수동 판정을 선택하세요.")

        st.markdown("#### 산출기준")
        rules = criterion.get("grade_rules", [])
        if rules:
            _safe_markdown_table(
                [
                    {
                        "등급": rule.get("grade"),
                        "원문 기준": rule.get("original_text") or rule.get("condition"),
                        "가중치": rule.get("weight"),
                    }
                    for rule in rules
                ],
            )
        else:
            st.write("표준 등급표가 없는 항목입니다.")

        st.markdown("#### 참고자료 및 제출서류")
        reference_rows = _reference_document_rows(criterion)
        if reference_rows:
            _safe_markdown_table(reference_rows)
        else:
            st.caption("G-SEED DB에 별도 참고자료/제출서류가 구조화되어 있지 않습니다.")

        st.markdown("#### 현재 근거")
        evidence = row.get("evidence", [])
        if evidence:
            _safe_markdown_table(
                [
                    {
                        "변수": item.get("variable"),
                        "값": item.get("value"),
                        "문서": item.get("source_document"),
                        "페이지": item.get("source_page"),
                        "근거": item.get("source_text"),
                    }
                    for item in evidence
                ],
            )
        else:
            st.caption("현재 연결된 근거가 없습니다.")

        st.markdown("#### 수동 판정")
        manual_options = _manual_options_for_criterion(criterion)
        current_option = _current_manual_option(row, criterion)
        cols = st.columns(len(manual_options))
        for idx, option in enumerate(manual_options):
            col = cols[idx]
            button_type = "primary" if option == current_option else "secondary"
            button_label = f"✓ {option}" if option == current_option else option
            if col.button(button_label, key=f"manual_{criterion_id}_{option}", type=button_type):
                st.session_state.review_overrides[criterion_id] = option
                st.session_state.review_target = None
                st.rerun()

        if st.button("닫기", key=f"close_{criterion_id}"):
            st.session_state.review_target = None
            st.rerun()


def _apply_manual_overrides(result: dict[str, Any]) -> dict[str, Any]:
    reviewed = deepcopy(result)
    overrides = st.session_state.review_overrides
    if not overrides:
        return reviewed

    existing_ids = {row.get("criterion_id") for row in reviewed.get("criteria_results", [])}
    for criterion_id, option in overrides.items():
        if criterion_id not in existing_ids:
            reviewed.setdefault("criteria_results", []).append(_empty_result_row(criterion_id))
            existing_ids.add(criterion_id)

    for row in reviewed.get("criteria_results", []):
        option = overrides.get(row.get("criterion_id"))
        if not option:
            continue
        _apply_one_override(row, option)

    _recalculate_summary(reviewed)
    return reviewed


def _apply_one_override(row: dict[str, Any], option: str) -> None:
    max_score = float(row.get("max_score") or 0)
    grade, weight = _parse_manual_grade_option(option, row.get("criterion_id"))
    if grade:
        row["status"] = "pass"
        row["grade"] = grade
        row["score"] = round(max_score * weight, 2)
        row["reason"] = "사용자 수동 검토로 충족 판정했습니다."
        row["manual_override"] = option
    elif option == "미충족":
        row["status"] = "fail"
        row["grade"] = None
        row["score"] = 0
        row["reason"] = "사용자 수동 검토로 미충족 판정했습니다."
        row["manual_override"] = option
    else:
        row["status"] = "insufficient_evidence"
        row["grade"] = None
        row["score"] = None
        row["reason"] = "사용자 수동 검토로 판단불가 처리했습니다."
        row["manual_override"] = option


def _recalculate_summary(result: dict[str, Any]) -> None:
    rows = _all_criteria_rows(result)
    raw_score = sum(r.get("score") or 0 for r in rows if r.get("status") in {"pass", "needs_review"})
    total_reference = result.get("summary", {}).get("total_reference_score", 135.0) or 135.0
    estimated = raw_score / total_reference * 100
    target = result.get("summary", {}).get("target_score", config.DEFAULT_TARGET_SCORE)
    result["summary"]["raw_estimated_score"] = round(raw_score, 2)
    result["summary"]["estimated_score"] = round(estimated, 2)
    result["summary"]["score_gap"] = round(max(0.0, target - estimated), 2)
    result["summary"].update(_grade_info_for_score(result, estimated))
    result["status_counts"] = _status_counts(rows)
    result["coverage_summary"] = _coverage_summary(rows)


def _grade_info_for_score(result: dict[str, Any], score: float) -> dict[str, Any]:
    """수동 검토 후 바뀐 점수에 맞춰 현재/다음 등급을 다시 계산한다."""
    case_key = result.get("summary", {}).get("certification_case", config.DEFAULT_CERTIFICATION_CASE)
    case = config.CERTIFICATION_GRADE_THRESHOLDS.get(case_key, config.CERTIFICATION_GRADE_THRESHOLDS[config.DEFAULT_CERTIFICATION_CASE])
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
        "estimated_certification_grade": current_grade,
        "next_certification_grade": next_grade,
        "next_grade_threshold": next_threshold,
        "next_grade_gap": None if next_threshold is None else round(max(0.0, next_threshold - score), 2),
    }


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    statuses = ["pass", "fail", "needs_review", "insufficient_evidence", "not_evaluated", "unknown"]
    return {status: sum(1 for row in rows if row.get("status") == status) for status in statuses}


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [r for r in rows if r.get("status") in {"pass", "fail", "needs_review"}]
    return {
        "evaluated_count": len(evaluable),
        "needs_review_count": sum(1 for r in rows if r.get("status") == "needs_review"),
        "insufficient_evidence_count": sum(1 for r in rows if r.get("status") == "insufficient_evidence"),
        "not_evaluated_count": sum(1 for r in rows if r.get("status") == "not_evaluated"),
        "extraction_suspect_count": sum(1 for r in rows if r.get("missing_variables")),
        "evaluated_raw_score": round(sum(r.get("score") or 0 for r in evaluable), 2),
        "evaluated_possible_score": round(sum(r.get("max_score") or 0 for r in evaluable), 2),
    }


def _coverage_metrics(result: dict[str, Any]) -> None:
    coverage = result.get("coverage_summary", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("평가 완료", coverage.get("evaluated_count", 0))
    c2.metric("검토 필요", coverage.get("needs_review_count", 0))
    c3.metric("근거 부족", coverage.get("insufficient_evidence_count", 0))
    c4.metric("판단 제외", coverage.get("not_evaluated_count", 0))
    c5.metric("추출 실패 의심", coverage.get("extraction_suspect_count", 0))


def _display_status(row: dict[str, Any]) -> str:
    mapping = {
        "pass": "충족",
        "needs_review": "검토 필요",
        "fail": "미충족",
        "insufficient_evidence": "근거 부족",
        "not_evaluated": "평가 제외",
        "unknown": "알 수 없음",
    }
    return mapping.get(row.get("status"), row.get("status", "-"))


def _overall_classification(result: dict[str, Any]) -> dict[str, Any]:
    kb = _kb()
    criteria = kb.get("criteria_index", {})
    result_by_id = {row.get("criterion_id"): row for row in _all_criteria_rows(result)}
    buckets = {"satisfied": [], "needs_more": [], "failed": [], "missing": []}

    for cid, criterion in criteria.items():
        row = result_by_id.get(cid)
        item = {"criterion_id": cid, "name": criterion.get("name")}
        if row is None:
            buckets["missing"].append(item)
        elif row.get("status") == "pass":
            buckets["satisfied"].append(item)
        elif row.get("status") in {"needs_review", "insufficient_evidence"}:
            buckets["needs_more"].append(item)
        elif row.get("status") == "fail":
            buckets["failed"].append(item)
        else:
            buckets["missing"].append(item)

    return {"total": len(criteria), **buckets}


def _kb() -> dict[str, Any]:
    if st.session_state.kb is None:
        st.session_state.kb = load_or_build_gseed_kb()
    return st.session_state.kb


def _answer_chat(prompt: str, kb: dict[str, Any], result: dict[str, Any] | None, llm: LLMClient) -> str:
    graph_retriever = GSeedGraphRetriever(kb, result, max_hops=2)
    rag_context = graph_retriever.retrieve(prompt, limit=4)
    project_context = _project_context(result)

    fallback = _fallback_chat_answer(prompt, rag_context, project_context)
    if not llm.enabled:
        return fallback

    response = llm.complete_json(
        system=(
            "너는 녹색건축 인증기준(G-SEED) 인증 항목 컨설팅 AI Agent다. "
            "반드시 제공된 graph RAG context와 현재 프로젝트 결과만 근거로 답한다. "
            "점수 계산은 추측하지 말고, 근거가 부족하면 필요한 문서/설계값/확인 방법을 분리해서 제안한다. "
            "START 노드를 primary criterion으로 보고, 그래프 경로의 필요 문서, 확인값, 등급 기준, 현재 프로젝트 근거를 우선 사용한다. "
            "질문과 관련성이 낮은 다른 항목은 나열하지 않는다. "
            "변수명 대신 사용자가 문서에서 확인할 수 있는 표현으로 말한다. "
            "답변은 1) 현재 판단, 2) 필요한 문서와 확인 내용, 3) 다음 행동 순서로 간결하게 작성한다."
        ),
        user=json.dumps(
            {
                "question": prompt,
                "graph_rag_context": rag_context,
                "project_context": project_context,
                "instruction": (
                    "사용자가 특정 항목을 물으면 START 노드의 항목명, 제출서류, 현재 프로젝트 상태를 연결해서 설명해. "
                    "필요 변수라는 표현은 쓰지 말고, '문서에서 확인할 내용'으로 바꿔 말해. "
                    "개선안을 물으면 문서 보완형/설계 개선형/비용 대비 효과형으로 나눠 제안해. "
                    "질문과 관련 없는 항목 번호는 답변에 포함하지 마."
                ),
            },
            ensure_ascii=False,
        ),
        fallback={"answer": fallback},
    )
    return response.get("answer", fallback)


def _project_context(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    return {
        "summary": result.get("summary", {}),
        "coverage_summary": result.get("coverage_summary", {}),
        "status_counts": result.get("status_counts", {}),
        "missing_documents": result.get("missing_documents", []),
        "positive_items": [
            {
                "criterion_id": row.get("criterion_id"),
                "name": row.get("name"),
                "status": row.get("status"),
                "grade": row.get("grade"),
                "score": row.get("score"),
            }
            for row in result.get("criteria_results", [])
            if row.get("status") in {"pass", "needs_review"}
        ],
    }


def _fallback_chat_answer(prompt: str, rag_context: dict[str, Any], project_context: dict[str, Any]) -> str:
    start_nodes = rag_context.get("start_nodes") or []
    if not start_nodes:
        return "관련 G-SEED 항목을 찾지 못했어. 항목 번호나 키워드를 조금 더 구체적으로 입력해줘."

    primary = start_nodes[0]

    lines = [f"가장 관련 있는 기준은 {primary.get('label')}이야."]
    if rag_context.get("context_text"):
        lines.append(rag_context["context_text"])

    if project_context:
        summary = project_context.get("summary", {})
        lines.append(
            f"\n현재 프로젝트 예상 점수는 {summary.get('estimated_score', 0):.2f}점이고, "
            f"목표까지 {summary.get('score_gap', 0):.2f}점이 부족해."
        )
        missing = project_context.get("missing_documents", [])
        if missing:
            lines.append("우선 보완 문서는 다음 묶음이 좋아.")
            for doc in missing[:3]:
                lines.append(f"- {doc.get('suggested_document')}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
