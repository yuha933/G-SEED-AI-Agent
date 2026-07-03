from __future__ import annotations

import argparse

from .agent import GSeedAgent, load_registry
from .config import DATA_DIR, DEFAULT_CERTIFICATION_CASE, DEFAULT_TARGET_GRADE, DEFAULT_TARGET_SCORE


def main() -> None:
    parser = argparse.ArgumentParser(description="G-SEED 사전검토 Agent")
    parser.add_argument("--registry", default=str(DATA_DIR / "document_registry.json"), help="문서 레지스트리 JSON 경로")
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE, help="목표 점수")
    parser.add_argument("--target-grade", default=DEFAULT_TARGET_GRADE, choices=["일반", "우량", "우수", "최우수"], help="목표 인증 등급")
    parser.add_argument("--certification-case", default=DEFAULT_CERTIFICATION_CASE, help="인증등급 점수기준 유형")
    parser.add_argument("--use-ocr", action="store_true", help="이미지형 PDF에 OCR 적용")
    parser.add_argument("--ocr-max-pages", type=int, default=5, help="문서별 OCR 최대 페이지 수")
    parser.add_argument("--run-name", default=None, help="실행 결과 폴더명")
    args = parser.parse_args()

    docs = load_registry(args.registry)
    agent = GSeedAgent(
        use_ocr=args.use_ocr,
        target_score=args.target_score,
        ocr_max_pages=args.ocr_max_pages,
        certification_case=args.certification_case,
        target_grade=args.target_grade,
    )
    result = agent.run(docs, run_name=args.run_name)

    summary = result["summary"]
    print(f"현재 예상 점수: {summary['estimated_score']}점 ({summary.get('score_basis', '점수')})")
    print(f"인증 기준 유형: {summary.get('certification_case_label')}")
    print(f"현재 예상 등급: {summary.get('estimated_certification_grade')}")
    print(f"목표 등급: {summary.get('target_grade')} / 목표 점수: {summary['target_score']}점")
    print(f"목표까지 부족 점수: {summary['score_gap']}점")
    if summary.get("next_certification_grade"):
        print(
            f"다음 등급({summary['next_certification_grade']}, "
            f"{summary['next_grade_threshold']}점)까지 부족 점수: {summary['next_grade_gap']}점"
        )
    print(f"원점수: {summary.get('raw_estimated_score')} / {summary.get('total_reference_score')}점")
    print("결과 폴더:", args.run_name or "자동 생성된 runs 폴더")


if __name__ == "__main__":
    main()
