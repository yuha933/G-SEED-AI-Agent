from __future__ import annotations

import json
import os
from typing import Any


class LLMClient:
    """OpenAI Responses API를 감싸는 간단한 LLM 클라이언트."""

    def __init__(self, model: str | None = None) -> None:
        # API Key는 코드에 직접 넣지 않고 환경변수에서 읽는다.
        self.api_key = os.getenv("OPENAI_API_KEY")

        # 모델도 환경변수로 바꿀 수 있게 둔다.
        # 예: $env:OPENAI_MODEL="gpt-5.4"
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-5.4-mini"

    @property
    def enabled(self) -> bool:
        """API Key가 있으면 LLM 호출을 활성화한다."""
        return bool(self.api_key)

    def complete_json(self, system: str, user: str, fallback: dict[str, Any]) -> dict[str, Any]:
        """JSON 답변을 생성하고 dict로 파싱한다."""
        if not self.enabled:
            return fallback

        json_system = (
            f"{system}\n\n"
            "응답은 반드시 유효한 JSON 객체 하나만 반환해. "
            "마크다운 코드블록, 설명문, 주석은 절대 붙이지 마."
        )

        try:
            response = self._create_response(system=json_system, user=user)
            content = self._extract_text(response)
            return self._parse_json_object(content, fallback)
        except Exception:
            return fallback

    def write_summary(self, raw_result: dict[str, Any]) -> str:
        """점수 계산 결과를 사용자가 읽기 쉬운 한국어 문장으로 요약한다."""
        fallback = self._fallback_summary(raw_result)
        if not self.enabled:
            return fallback

        response = self.complete_json(
            system=(
                "너는 녹색건축 인증기준(G-SEED) 사전검토 결과를 설명하는 컨설팅 AI Agent다. "
                "제공된 결과 JSON만 근거로 하며, 근거 없는 점수나 등급을 새로 만들지 않는다."
            ),
            user=(
                "다음 결과를 바탕으로 사용자가 바로 이해할 수 있는 요약문을 작성해.\n"
                "반드시 summary 필드 하나를 가진 JSON으로 답해.\n"
                "포함할 내용: 현재 예상 점수, 현재 예상 등급, 다음 등급까지 부족 점수, "
                "근거가 부족하거나 사람 검토가 필요한 항목 수, 다음 확인 방향.\n\n"
                + json.dumps(raw_result, ensure_ascii=False)
            ),
            fallback={"summary": fallback},
        )
        return response.get("summary", fallback)

    def _create_response(self, system: str, user: str) -> Any:
        """OpenAI Responses API를 호출한다."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        return client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Responses API 응답에서 텍스트를 최대한 안전하게 꺼낸다."""
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()

        # SDK 버전에 따라 output_text가 없을 때를 대비한 보조 파서.
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()

    @staticmethod
    def _parse_json_object(content: str, fallback: dict[str, Any]) -> dict[str, Any]:
        """LLM 출력에서 JSON 객체를 파싱한다."""
        if not content:
            return fallback

        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else fallback
        except json.JSONDecodeError:
            pass

        # 혹시 앞뒤에 문장이 붙은 경우 첫 { ... } 구간만 다시 시도한다.
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return fallback

        try:
            parsed = json.loads(content[start : end + 1])
            return parsed if isinstance(parsed, dict) else fallback
        except json.JSONDecodeError:
            return fallback

    @staticmethod
    def _fallback_summary(raw_result: dict[str, Any]) -> str:
        """LLM이 없거나 실패했을 때 사용할 기본 요약문."""
        summary = raw_result.get("summary", {})
        estimated = summary.get("estimated_score", 0)
        target = summary.get("target_score", 0)
        gap = summary.get("score_gap", 0)
        target_grade = summary.get("target_grade") or "목표 등급"
        current_grade = summary.get("estimated_certification_grade") or "등급 없음"
        case_label = summary.get("certification_case_label") or "해당 인증 유형"
        raw = summary.get("raw_estimated_score", 0)
        total = summary.get("total_reference_score", 0)
        coverage = raw_result.get("coverage_summary", {})

        return (
            f"현재 입력 문서 기준으로 확인 가능한 예상 점수는 {estimated:.2f}점입니다. "
            f"{case_label} 기준 현재 예상 등급은 '{current_grade}'이며, "
            f"목표 등급 '{target_grade}'의 커트라인 {target:.2f}점까지 {gap:.2f}점이 부족합니다. "
            f"원점수 기준으로는 {raw:.2f}/{total:.2f}점이 확인되었습니다. "
            f"현재 문서에서 평가 완료된 항목은 {coverage.get('evaluated_count', 0)}개, "
            f"사람 검토가 필요한 항목은 {coverage.get('needs_review_count', 0)}개, "
            f"근거 일부 부족 항목은 {coverage.get('insufficient_evidence_count', 0)}개, "
            f"추출 실패 의심 항목은 {coverage.get('extraction_suspect_count', 0)}개입니다. "
            "따라서 이 결과는 최종 인증 점수라기보다, 현재 문서로 어디까지 판단 가능한지 보여주는 "
            "사전검토 결과로 해석해야 합니다."
        )
