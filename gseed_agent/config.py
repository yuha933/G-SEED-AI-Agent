from pathlib import Path


# project 폴더 기준 경로
PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent

# G-SEED 기준 DB 경로
GSEED_DIR = PROJECT_DIR / "G-SEED_주거용건축물"

# 실행 결과와 입력 문서 레지스트리 경로
RUNS_DIR = PROJECT_DIR / "gseed_agent" / "runs"
DATA_DIR = PROJECT_DIR / "gseed_agent" / "data"

# OCR 모델/캐시 경로
OCR_CACHE_DIR = PROJECT_DIR / ".ocr_cache_py313"

# 현재 프로젝트의 기본 평가 대상: 신축 주거용 건축물
DEFAULT_CERTIFICATION_CASE = "new_residential"
DEFAULT_TARGET_GRADE = "일반"

# 녹색건축 인증기준 별표 9: 인증등급별 점수기준
CERTIFICATION_GRADE_THRESHOLDS = {
    "new_residential": {
        "label": "신축 주거용",
        "grades": {
            "최우수": 74.0,
            "우수": 66.0,
            "우량": 58.0,
            "일반": 50.0,
        },
    },
    "new_detached_house": {
        "label": "신축 단독주택",
        "grades": {
            "최우수": 74.0,
            "우수": 66.0,
            "우량": 58.0,
            "일반": 50.0,
        },
    },
    "new_non_residential": {
        "label": "신축 비주거용",
        "grades": {
            "최우수": 80.0,
            "우수": 70.0,
            "우량": 60.0,
            "일반": 50.0,
        },
    },
    "existing_residential": {
        "label": "기존 주거용",
        "grades": {
            "최우수": 69.0,
            "우수": 61.0,
            "우량": 53.0,
            "일반": 45.0,
        },
    },
    "existing_non_residential": {
        "label": "기존 비주거용",
        "grades": {
            "최우수": 75.0,
            "우수": 65.0,
            "우량": 55.0,
            "일반": 45.0,
        },
    },
    "green_remodeling_residential": {
        "label": "그린리모델링 주거용",
        "grades": {
            "최우수": 69.0,
            "우수": 61.0,
            "우량": 53.0,
            "일반": 45.0,
        },
    },
    "green_remodeling_non_residential": {
        "label": "그린리모델링 비주거용",
        "grades": {
            "최우수": 75.0,
            "우수": 65.0,
            "우량": 55.0,
            "일반": 45.0,
        },
    },
}

# 기본 목표 점수는 기본 평가 대상의 일반 등급 기준
DEFAULT_TARGET_SCORE = CERTIFICATION_GRADE_THRESHOLDS[DEFAULT_CERTIFICATION_CASE]["grades"][DEFAULT_TARGET_GRADE]

# 현재 실험은 서로 다른 건물의 문서를 하나로 합쳐 검증하지 않는다.
DOCUMENT_INDEPENDENT_MODE = True
