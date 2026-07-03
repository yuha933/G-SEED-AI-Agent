from __future__ import annotations

import os
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz

from . import config


@dataclass
class ParsedPage:
    page: int
    text: str
    text_length: int
    image_count: int
    title: str | None = None
    text_blocks: list[str] | None = None
    table_count: int = 0
    table_text: list[str] | None = None
    ocr_used: bool = False
    needs_ocr: bool = False
    ocr_error: str | None = None


@dataclass
class ParsedDocument:
    document_id: str
    path: str
    filename: str
    page_count: int
    text_page_count: int
    image_only_page_count: int
    document_type: str
    pages: list[ParsedPage]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pages"] = [asdict(p) for p in self.pages]
        return data


_OCR_ENGINE: Any | None = None
_OCR_INIT_ERROR: str | None = None


def read_pdf(
    path: str | Path,
    document_id: str | None = None,
    use_ocr: bool = False,
    ocr_max_pages: int | None = 5,
    ocr_dpi: int = 160,
) -> ParsedDocument:
    """PDF를 페이지별 텍스트로 읽는다.

    일반 텍스트가 없거나 너무 적은 이미지형 페이지는 OCR 대상이다.
    실험 단계에서는 처리 시간을 줄이기 위해 OCR 페이지 수를 제한한다.
    """
    pdf_path = Path(path)
    doc = fitz.open(str(pdf_path))
    pages: list[ParsedPage] = []
    ocr_count = 0
    planned_ocr_pages = _planned_ocr_pages(len(doc), ocr_max_pages) if use_ocr else set()

    for idx, page in enumerate(doc, start=1):
        base_text = page.get_text("text").strip()
        image_count = len(page.get_images(full=True))
        text_blocks = _extract_text_blocks(page)
        table_text = _extract_table_text(page, base_text)
        title = _guess_page_title(base_text, text_blocks)
        text = _merge_text_sources(base_text, table_text)
        original_needs_ocr = _should_try_ocr(text=text, image_count=image_count)
        ocr_used = False
        ocr_error = None

        can_ocr = use_ocr and original_needs_ocr
        if planned_ocr_pages:
            can_ocr = can_ocr and idx in planned_ocr_pages
        elif ocr_max_pages is not None and ocr_count >= ocr_max_pages:
            can_ocr = False

        if can_ocr:
            ocr_text, ocr_error = _try_ocr_page(page, dpi=ocr_dpi)
            if ocr_text.strip():
                text = _merge_text_sources(text, [f"[OCR]\n{ocr_text}"])
            ocr_used = True
            ocr_count += 1

        needs_ocr = original_needs_ocr and len(text.strip()) == 0
        pages.append(
            ParsedPage(
                page=idx,
                text=text,
                text_length=len(text),
                image_count=image_count,
                title=title,
                text_blocks=text_blocks,
                table_count=len(table_text),
                table_text=table_text,
                ocr_used=ocr_used,
                needs_ocr=needs_ocr,
                ocr_error=ocr_error,
            )
        )

    text_page_count = sum(1 for p in pages if p.text_length > 20)
    image_only_page_count = sum(1 for p in pages if p.needs_ocr)
    sample_text = "\n".join(p.text[:500] for p in pages[:10])

    return ParsedDocument(
        document_id=document_id or pdf_path.stem,
        path=str(pdf_path.resolve()),
        filename=pdf_path.name,
        page_count=len(doc),
        text_page_count=text_page_count,
        image_only_page_count=image_only_page_count,
        document_type=classify_document(pdf_path.name, sample_text),
        pages=pages,
    )


def _should_try_ocr(text: str, image_count: int) -> bool:
    """OCR 적용 여부를 판단한다.

    기존에는 텍스트가 0인 페이지만 OCR했지만, 실제 문서는 이미지 표 위에
    작은 텍스트만 얹혀 있는 경우가 많아 짧은 텍스트 페이지도 OCR 대상으로 둔다.
    """
    if image_count <= 0:
        return False
    compact = text.strip()
    if not compact:
        return True
    return len(compact) < 80


def _planned_ocr_pages(page_count: int, max_pages: int | None) -> set[int]:
    """긴 이미지 PDF에서 앞 페이지만 OCR하지 않도록 페이지를 분산 선택한다."""
    if max_pages is None or max_pages >= page_count:
        return set(range(1, page_count + 1))
    if max_pages <= 0:
        return set()

    selected: set[int] = set()

    # 앞쪽에는 문서 개요/주요 표가 있는 경우가 많아 일부를 유지한다.
    head_count = min(10, max_pages)
    selected.update(range(1, min(page_count, head_count) + 1))

    remaining = max_pages - len(selected)
    if remaining <= 0:
        return selected

    # 뒤쪽 도면/부록에도 산출표가 있을 수 있어 일부를 확보한다.
    tail_count = min(5, remaining)
    selected.update(range(max(1, page_count - tail_count + 1), page_count + 1))

    remaining = max_pages - len(selected)
    if remaining <= 0:
        return selected

    # 남은 수량은 문서 전체에 균등 분산한다.
    if remaining == 1:
        selected.add(max(1, page_count // 2))
        return selected

    for i in range(remaining):
        page_no = 1 + round(i * (page_count - 1) / max(1, remaining - 1))
        selected.add(page_no)

    # 중복으로 개수가 부족하면 앞에서부터 빈 페이지 번호를 채운다.
    cursor = 1
    while len(selected) < max_pages and cursor <= page_count:
        selected.add(cursor)
        cursor += 1
    return selected


def _extract_text_blocks(page: fitz.Page) -> list[str]:
    """페이지의 텍스트 블록을 위에서 아래 순서로 추출한다."""
    blocks: list[tuple[float, float, str]] = []
    try:
        for block in page.get_text("blocks") or []:
            if len(block) < 5:
                continue
            x0, y0, text = float(block[0]), float(block[1]), str(block[4]).strip()
            if text:
                blocks.append((y0, x0, text))
    except Exception:
        return []
    return [text for _, _, text in sorted(blocks)]


def _extract_table_text(page: fitz.Page, text: str) -> list[str]:
    """PyMuPDF의 표 탐지 결과를 텍스트 행으로 변환한다.

    표 탐지는 페이지에 따라 매우 느릴 수 있어서 표/산출서 가능성이 있는
    페이지에서만 제한적으로 수행한다.
    """
    rows: list[str] = []
    if os.getenv("GSEED_ENABLE_PYMUPDF_TABLES", "0") != "1":
        return _fast_table_like_lines(text)

    if not _looks_like_table_page(text):
        return rows

    try:
        finder = page.find_tables()
    except Exception:
        return rows

    for table_idx, table in enumerate(getattr(finder, "tables", []) or [], start=1):
        try:
            extracted = table.extract()
        except Exception:
            continue
        if not extracted:
            continue
        rows.append(f"[TABLE {table_idx}]")
        for row in extracted:
            cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
            if cells:
                rows.append(" | ".join(cells))
    return rows


def _fast_table_like_lines(text: str) -> list[str]:
    """무거운 표 인식 대신 표처럼 보이는 행을 빠르게 보존한다."""
    rows: list[str] = []
    keywords = ["구분", "합계", "면적", "비율", "등급", "점수", "산출", "제품명", "제조사", "생태면적률", "에너지"]
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if any(keyword in line for keyword in keywords) and re.search(r"\d", line):
            rows.append(line)
    if rows:
        return ["[TABLE-LIKE LINES]"] + rows[:40]
    return []


def _looks_like_table_page(text: str) -> bool:
    """표 탐지를 시도할 만한 페이지인지 가볍게 판단한다."""
    if not text.strip():
        return False
    keywords = [
        "구분",
        "합계",
        "면적",
        "비율",
        "등급",
        "점수",
        "산출",
        "계획",
        "적용",
        "수량",
        "제품명",
        "제조사",
        "생태면적률",
        "에너지",
        "자재",
    ]
    hit_count = sum(1 for keyword in keywords if keyword in text)
    return hit_count >= 2 or text.count("\n") >= 12


def _guess_page_title(text: str, blocks: list[str]) -> str | None:
    """페이지 제목처럼 보이는 첫 줄을 가볍게 추정한다."""
    candidates: list[str] = []
    for source in blocks[:3] or [text]:
        candidates.extend(line.strip() for line in source.splitlines() if line.strip())
    for line in candidates:
        if 2 <= len(line) <= 80:
            return line
    return None


def _merge_text_sources(base_text: str, extra_sections: list[str]) -> str:
    """본문, 표, OCR 텍스트를 중복을 줄이며 하나의 분석 텍스트로 합친다."""
    parts = [base_text.strip()] if base_text.strip() else []
    for section in extra_sections:
        section = section.strip()
        if section and section not in parts:
            parts.append(section)
    return "\n".join(parts).strip()


def classify_document(filename: str, sample_text: str) -> str:
    """파일명과 초반 텍스트로 문서 유형을 가볍게 분류한다."""
    name = filename.lower()
    text = sample_text.lower()

    if "마감" in filename or "자재" in filename or "제품명" in sample_text:
        return "material"
    if "에너지" in filename or "energy" in name or "에너지사용" in sample_text:
        return "energy"
    if "조경" in filename or "생태" in sample_text or "식재" in sample_text:
        return "landscape"
    if "시공" in filename or "시공계획" in sample_text:
        return "construction"
    if "도면" in filename or "평면" in sample_text or "배치도" in sample_text:
        return "drawing"
    return "unknown"


def _prepare_ocr_environment() -> None:
    """PaddleOCR/PaddleX 캐시를 프로젝트 내부로 돌려 권한 문제를 피한다."""
    cache_dir = config.OCR_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    pseudo_home = cache_dir / "home"
    pseudo_home.mkdir(parents=True, exist_ok=True)

    # 일부 PaddleX 버전은 사용자 홈의 .paddlex에 접근한다.
    # 현재 프로세스 안에서만 홈/캐시 위치를 바꿔 안전하게 초기화한다.
    os.environ["HOME"] = str(pseudo_home)
    os.environ["USERPROFILE"] = str(pseudo_home)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir / "paddlex")
    os.environ["PADDLEX_HOME"] = str(cache_dir / "paddlex")
    os.environ["PADDLE_HOME"] = str(cache_dir / "paddle")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    os.environ["PADDLEOCR_HOME"] = str(cache_dir / "paddleocr")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_onednn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"


def _get_ocr_engine() -> Any:
    """OCR 엔진은 초기화 비용이 커서 한 번만 만든다."""
    global _OCR_ENGINE, _OCR_INIT_ERROR

    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    if _OCR_INIT_ERROR:
        raise RuntimeError(_OCR_INIT_ERROR)

    try:
        _prepare_ocr_environment()
        from paddleocr import PaddleOCR

        # 문서 방향/왜곡 보정 모델은 무겁고 Windows oneDNN 오류가 날 수 있어 비활성화한다.
        _OCR_ENGINE = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        return _OCR_ENGINE
    except Exception as exc:  # pragma: no cover - 설치 환경 의존
        _OCR_INIT_ERROR = str(exc)
        raise


def _try_ocr_page(page: fitz.Page, dpi: int = 160) -> tuple[str, str | None]:
    """단일 페이지를 이미지로 렌더링한 뒤 OCR 텍스트를 반환한다."""
    errors: list[str] = []
    try:
        pix = page.get_pixmap(dpi=dpi)

        if os.name == "nt":
            lines, error = _try_windows_ocr(pix)
            if lines:
                return "\n".join(lines), None
            if error:
                errors.append(f"WindowsOCR: {error}")

        image = _pixmap_to_array(pix)
        try:
            ocr = _get_ocr_engine()
            result = _run_ocr(ocr, image)
            lines = _extract_ocr_lines(result)
            if lines:
                return "\n".join(lines), None
        except Exception as exc:
            errors.append(f"PaddleOCR: {exc}")

        try:
            lines = _try_rapidocr(image)
            if lines:
                return "\n".join(lines), None
        except Exception as exc:
            errors.append(f"RapidOCR: {exc}")

        return "", " | ".join(errors) if errors else "OCR 결과 없음"
    except Exception as exc:
        return "", str(exc)


def _pixmap_to_array(pix: fitz.Pixmap) -> Any:
    """PyMuPDF 픽스맵을 PaddleOCR이 읽을 수 있는 numpy 이미지로 바꾼다."""
    import numpy as np

    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = image[:, :, :3]
    if pix.n == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def _run_ocr(ocr: Any, image_path: str) -> Any:
    """PaddleOCR 2.x/3.x API 차이를 흡수한다."""
    if hasattr(ocr, "predict"):
        return ocr.predict(image_path)
    return ocr.ocr(image_path, cls=True)


def _try_windows_ocr(pix: fitz.Pixmap) -> tuple[list[str], str | None]:
    """Windows 내장 OCR 엔진을 사용한다. 한국어 언어팩이 있으면 한글 인식 품질이 좋다."""
    temp_dir = config.OCR_CACHE_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_path = temp_dir / f"ocr_{uuid.uuid4().hex}.png"
    script_path = temp_dir / "windows_ocr.ps1"

    try:
        pix.save(str(image_path))
        if not script_path.exists():
            script_path.write_text(_WINDOWS_OCR_SCRIPT, encoding="utf-8")

        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                str(image_path.resolve()),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        if completed.returncode != 0:
            return [], completed.stderr.strip() or completed.stdout.strip()

        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return lines, None
    except Exception as exc:
        return [], str(exc)
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass


def _try_rapidocr(image: Any) -> list[str]:
    """PaddleOCR 실패 시 ONNX 기반 RapidOCR을 보조 OCR로 사용한다."""
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    result, _ = engine(image)
    return [str(row[1]).strip() for row in result or [] if len(row) >= 2 and str(row[1]).strip()]


_WINDOWS_OCR_SCRIPT = r"""
param([string]$ImagePath)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Foundation, ContentType=WindowsRuntime]

function AwaitOp($op, [type]$resultType) {
  $methods = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 }
  $method = $methods |
    Where-Object { $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } |
    Select-Object -First 1
  $task = $method.MakeGenericMethod($resultType).Invoke($null, @($op))
  $task.Wait()
  return $task.Result
}

$file = AwaitOp ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
$stream = AwaitOp ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = AwaitOp ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = AwaitOp ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$language = [Windows.Globalization.Language]::new('ko')
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) {
  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
}
if ($null -eq $engine) {
  throw '사용 가능한 Windows OCR 엔진을 찾지 못했습니다.'
}

$result = AwaitOp ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$result.Lines | ForEach-Object {
  ($_.Words | ForEach-Object { $_.Text }) -join ' '
}
"""


def _extract_ocr_lines(result: Any) -> list[str]:
    """PaddleOCR 결과 객체에서 인식된 문자열만 꺼낸다."""
    lines: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text:
            lines.append(text)

    def walk(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, dict):
            if "rec_texts" in obj:
                for text in obj.get("rec_texts") or []:
                    add(text)
                return
            if "text" in obj:
                add(obj.get("text"))
            for value in obj.values():
                walk(value)
            return
        if isinstance(obj, (list, tuple)):
            # PaddleOCR 2.x: [box, (text, score)] 형태
            if len(obj) >= 2 and isinstance(obj[1], (list, tuple)) and obj[1]:
                if isinstance(obj[1][0], str):
                    add(obj[1][0])
                    return
            for value in obj:
                walk(value)

    walk(result)
    return lines
