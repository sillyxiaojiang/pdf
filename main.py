import asyncio
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz

from dispatcher import DocumentClassifier, QualityGate
from engines import EngineImage, EngineVLM
from ingest_mineru_md import build_device_params, collect_mineru_file_links
from mapper import MaterialMapper, is_valid_device_model
from mineru_client import DEFAULT_MINERU_SERVICE_URL, MinerUClient
from models import DmLoader, DeviceMain, DeviceParam, FileLink, generate_fingerprint
from settings import load_settings
from text_utils import normalize_text

settings = load_settings()

INCOMING_DIR = settings.incoming_dir
DONE_DIR = settings.done_dir
IMG_DIR = settings.img_dir
DLQ_DIR = settings.dlq_dir
OCR_PENDING_DIR = settings.ocr_pending_dir
RETRY_DIR = settings.retry_dir

for directory in [INCOMING_DIR, DONE_DIR, IMG_DIR, DLQ_DIR, OCR_PENDING_DIR, RETRY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def _zh(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


def _same_path(path_a: Path, path_b: Path) -> bool:
    return path_a.resolve(strict=False) == path_b.resolve(strict=False)


def _available_target_path(target_dir: Path, file_name: str, current_path: Path) -> Path:
    candidate = target_dir / file_name
    if _same_path(candidate, current_path) or not candidate.exists():
        return candidate

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    index = 1
    while True:
        candidate = target_dir / f"{stem}__dup{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _cleanup_working_dir(label: str = "") -> int:
    working_root = IMG_DIR.resolve(strict=False)
    if not working_root.exists():
        return 0

    removed = 0
    for child in list(working_root.iterdir()):
        resolved_child = child.resolve(strict=False)
        if working_root not in resolved_child.parents:
            print(f"  Warning: skip unsafe cleanup target: {child}")
            continue

        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except Exception as exc:
            print(f"  Warning: failed to cleanup working artifact {child}: {exc}")

    if removed:
        suffix = f" ({label})" if label else ""
        print(f"  Cleaned {removed} working artifact root entries{suffix}")
    return removed


def _normalize_model_token(model_name: str) -> str:
    clean_name = (model_name or "").upper().strip()
    clean_name = clean_name.replace("_", "-")
    clean_name = clean_name.replace("/", "-")
    clean_name = re.sub(r"[\x00-\x1f]+", "-", clean_name)
    clean_name = re.sub(r"[()\[\]{}<>\uFF08\uFF09\u3010\u3011\u300A\u300B]", "-", clean_name)
    clean_name = re.sub(r"(?<=[A-Z0-9])\s+(?=[A-Z0-9])", "-", clean_name)
    clean_name = re.sub(r"\s*-\s*", "-", clean_name)
    clean_name = re.sub(r"\s+", "", clean_name)
    clean_name = re.sub(r"-{2,}", "-", clean_name)
    clean_name = clean_name.strip("-")
    clean_name = clean_name.rstrip(".,;:/\\")
    return clean_name


def _core_model(model_name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _normalize_model_token(model_name))


def _display_model_token(model_name: str) -> str:
    normalized = _normalize_model_token(model_name)
    pair_match = re.match(r"^(QTZ[-\s]?\d{2,6}[A-Z0-9.-]*)-(ZS[LC][-\s]?\d{2,6}[A-Z0-9.-]*)$", normalized, re.IGNORECASE)
    if pair_match:
        left = _normalize_model_token(pair_match.group(1))
        right = _normalize_model_token(pair_match.group(2))
        return f"{left}({right})"
    return normalized




def _sanitize_model_source_text(text: str) -> str:
    sanitized_text = re.sub(r"\.(?:PDF|DOCX?|XLSX?)\b", " ", text or "", flags=re.IGNORECASE)
    sanitized_text = sanitized_text.replace("_", " ")
    sanitized_text = sanitized_text.replace("/", "-")
    sanitized_text = re.sub(r"[\x00-\x1f]+", "-", sanitized_text)
    sanitized_text = re.sub(r"[()\[\]{}<>\uFF08\uFF09\u3010\u3011\u300A\u300B]", "-", sanitized_text)
    sanitized_text = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Z0-9])", " ", sanitized_text, flags=re.IGNORECASE)
    sanitized_text = re.sub(r"(?<=[A-Z0-9])(?=[\u4e00-\u9fff])", " ", sanitized_text, flags=re.IGNORECASE)
    sanitized_text = re.sub(r"(?<=[A-Z0-9])\s+(?=[A-Z0-9])", "-", sanitized_text, flags=re.IGNORECASE)
    sanitized_text = re.sub(r"\s+", " ", sanitized_text)
    sanitized_text = re.sub(r"-{2,}", "-", sanitized_text)
    return sanitized_text


def _expand_compound_model_token(model_name: str) -> list[str]:
    clean_name = _normalize_model_token(model_name)
    for pattern in COMPOUND_TOWER_MODEL_PATTERNS:
        match = pattern.match(clean_name)
        if not match:
            continue

        suffixes = [item for item in match.group("suffixes").split("-") if item]
        if len(suffixes) < 2:
            return [clean_name]

        return [f"{match.group('base').upper()}-{suffix.upper().rstrip('T')}T" for suffix in suffixes]

    return [clean_name]


def _split_fused_model_candidate(model_name: str) -> list[str]:
    clean_name = _normalize_model_token(model_name)
    split_candidates = [
        part.strip("-")
        for part in re.split(rf"-(?=(?:{SPLITTABLE_MODEL_PREFIXES})[-\s]?\d)", clean_name, flags=re.IGNORECASE)
        if part.strip("-")
    ]
    if len(split_candidates) <= 1 and len(clean_name) <= 20:
        return [clean_name]

    short_candidates = [part for part in split_candidates if 3 <= len(part) <= 20]
    return short_candidates or [clean_name]


def _extract_qtz_pair_candidates(raw_text: str, sanitized_text: str) -> tuple[list[str], set[str]]:
    pair_pattern = re.compile(
        r"(?P<qtz>QTZ[-\s]?\d{2,6}[A-Z0-9.-]*)\s*(?:\(|[（\[])?\s*(?P<zs>ZS[LC][-\s]?\d{2,6}[A-Z0-9.-]*)\s*(?:\)|[）\]])?",
        re.IGNORECASE,
    )
    pair_candidates: list[str] = []
    pair_components: set[str] = set()

    for source in [raw_text or "", sanitized_text]:
        for match in pair_pattern.finditer(source):
            qtz_model = _normalize_model_token(match.group("qtz"))
            zs_model = _normalize_model_token(match.group("zs"))
            pair_model = _normalize_model_token(f"{qtz_model}-{zs_model}")
            if not (is_valid_device_model(qtz_model) and is_valid_device_model(zs_model)):
                continue
            if not is_valid_device_model(pair_model):
                continue
            if pair_model not in pair_candidates:
                pair_candidates.append(pair_model)
            pair_components.add(qtz_model)
            pair_components.add(zs_model)

            # Also emit normalized display variants so the parser can map
            # QTZ1600A / QTZ1200A to the matched paired notation when needed.
            if qtz_model.endswith("A"):
                base = qtz_model[:-1]
                suffix = base[3:]
                for alias_pair in [f"{base}(ZSC{suffix})", f"{base}(ZSL{suffix})"]:
                    if alias_pair not in pair_candidates:
                        pair_candidates.append(alias_pair)

    return pair_candidates, pair_components


def extract_model_candidates(text: str) -> list[str]:
    if not text:
        return []

    sanitized_text = _sanitize_model_source_text(text)
    pair_candidates, pair_components = _extract_qtz_pair_candidates(text, sanitized_text)
    sanitized_text = re.sub(
        r"(?<=[0-9T])(?=(?:XCA|XCT|STC|STB|SSC|SPC|SAC|SPS|QY|QAY|ZTC|ZAT|ZCT|ZRT|ZCC|SCC|XGC|QUY|LR|LTM|CC|QTZ|QTD|QTP|TCT|TCR|STT|STL|SPT|WQ|ZSL|ZSC|JT|JTL|JTT|JTZ))",
        " ",
        sanitized_text,
        flags=re.IGNORECASE,
    )

    candidates: list[str] = list(pair_candidates)
    for pattern in MODEL_TOKEN_PATTERNS:
        for match in pattern.finditer(sanitized_text):
            for split_model in _split_fused_model_candidate(match.group(0)):
                for expanded_model in _expand_compound_model_token(split_model):
                    normalized_model = _normalize_model_token(expanded_model)
                    if normalized_model in pair_components:
                        continue
                    if is_valid_device_model(normalized_model) and normalized_model not in candidates:
                        candidates.append(normalized_model)
    return candidates


def choose_best_model_candidate(candidate_text: str, known_models: list[str], current_model: str) -> Optional[str]:
    candidates = extract_model_candidates(candidate_text)
    if not candidates:
        return None

    known_by_core = {_core_model(model): model for model in known_models}
    ranked_candidates: list[str] = []

    for candidate in candidates:
        candidate_core = _core_model(candidate)
        if candidate_core in known_by_core:
            ranked_candidates.append(known_by_core[candidate_core])
            continue

        matched_known_model = None
        for known_core, known_model in known_by_core.items():
            if candidate_core and (candidate_core in known_core or known_core in candidate_core):
                matched_known_model = known_model if len(known_model) >= len(candidate) else candidate
                break

        ranked_candidates.append(matched_known_model or candidate)

    ranked_candidates = list(dict.fromkeys(ranked_candidates))
    if current_model in ranked_candidates:
        return current_model

    ranked_candidates.sort(key=lambda item: (item in known_models, len(item)), reverse=True)
    return ranked_candidates[0] if ranked_candidates else None


def collect_page_model_targets(page_text: str, known_models: list[str], current_model: str, hinted_model: Optional[str] = None, vlm_model: Optional[str] = None) -> list[str]:
    candidates = extract_model_candidates(page_text)
    targets: list[str] = []

    def add(model: Optional[str]):
        if not model:
            return
        normalized = _normalize_model_token(model)
        matched = choose_best_model_candidate(normalized, known_models, current_model) or normalized
        if matched not in targets:
            targets.append(matched)

    add(current_model)
    add(hinted_model)
    add(vlm_model)
    for candidate in candidates:
        add(candidate)

    return targets


TITLE_SKIP_RE = re.compile("|".join([_zh(r"\u76ee\u5f55"), "Contents", "Index", _zh(r"\u524d\u8a00")]), re.IGNORECASE)
TITLE_KEYWORDS = [
    _zh(r"\u8bf4\u660e\u4e66"),
    _zh(r"\u624b\u518c"),
    _zh(r"\u6027\u80fd\u8868"),
    _zh(r"\u53c2\u6570\u8868"),
    _zh(r"\u6280\u672f\u53c2\u6570"),
    _zh(r"\u6280\u672f\u89c4\u683c"),
    _zh(r"\u64cd\u4f5c\u6307\u5357"),
    "Manual",
    "Data Sheet",
    "Specifications",
    _zh(r"\u8d77\u91cd\u673a"),
    "Crane",
]
SPLITTABLE_MODEL_PREFIXES = (
    r"XCA|XCT|STC|STB|SSC|SPC|SAC|SPS|QY|QAY|ZTC|ZAT|ZCT|ZRT|ZCC|SCC|XGC|QUY|LR|LTM|CC|"
    r"QTZ|QTD|QTP|TCT|TCR|STT|STL|ST|TC|JT|JTL|JTT|JTZ|SPT|WQ|CJ|ZSL|ZSC"
)
MODEL_TOKEN_PATTERNS = [
    re.compile(r"\b(?:XCA|XCT|STC|STB|SSC|SPC|SAC|SPS)[-\s]?\d{2,6}[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(r"\b(?:QY|QAY|ZTC|ZAT|ZCT|ZRT|ZCC)[-\s]?\d{2,6}[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(r"\b(?:SCC|XGC|QUY|LR)[-\s]?\d{2,6}[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(r"\bLTM[-\s]?\d{2,6}(?:[-.]\d+)*(?:[A-Z]\d*)?\b", re.IGNORECASE),
    re.compile(r"\bCC[-\s]?\d{2,6}[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(r"\b(?:QTZ|QTD|QTP|TCT|TCR|STT|STL|EC-H)[-\s]?\d{2,6}[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(r"\bTC[-\s]?\d{2,6}[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(r"\b(?:ZSL|ZSC)[-\s]?\d{2,6}[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(
        r"\bST\d{2,5}[A-Z]?(?:-\d{2,5}[A-Z]?)?(?:-\d+(?:\.\d+)?[A-Z]?)?(?:-\d+(?:\.\d+)?[A-Z]?)?(?:-\d+(?:\.\d+)?[A-Z]?)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d*HC-L\d*[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(r"\b(?:JT|JTL|JTT|JTZ)[-\w]*\d+\b", re.IGNORECASE),
    re.compile(r"\bSPT[-\s]?\d{2,5}[A-Z0-9.-]*\b", re.IGNORECASE),
    re.compile(r"\bWQ\d{1,2}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\b(?:\d{1,2}[-\s]?)?CJ[-\s]?\d{2,4}(?:[-\s]?\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bC\d{4}[A-Z]?(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bD\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bF[0O][-\s]?\d{2}[A-Z](?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bFL\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bEL\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bH\d{1,2}(?:[-.]\d{2,3}[A-Z]?)\b", re.IGNORECASE),
    re.compile(r"\bK\d{2}(?:[-.]\d{2,3}[A-Z]?)\b", re.IGNORECASE),
    re.compile(r"\bM\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bMC\d{2,4}[A-Z]?(?:[-\s](?:H|K|L|TL|LH)\d{1,2})?\b", re.IGNORECASE),
    re.compile(r"\bP\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bQD\d{1,2}[A-Z]?(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b", re.IGNORECASE),
    re.compile(r"\bR\d{2}(?:[-.]\d{2,3}[A-Z]?)\b", re.IGNORECASE),
    re.compile(r"\bS\d{2,4}(?:[-\s](?:E\d|G\d|H\d{1,2}|K\d{1,2}|L\d{1,2}|LK\d{1,2}|LL\d{1,2}|LH\d{1,2}|TL\d{1,2}))\b", re.IGNORECASE),
]
COMPOUND_TOWER_MODEL_PATTERNS = [
    re.compile(r"^(?P<base>(?:STT|STL|TCT|TCR|QTZ|QTD|QTP|TC)\d{2,6}[A-Z]?)(?P<suffixes>(?:-\d+(?:\.\d+)?T?){2,})$", re.IGNORECASE),
    re.compile(r"^(?P<base>ST\d{2,5}[A-Z]?(?:-\d{2,5}[A-Z]?)?)(?P<suffixes>(?:-\d+(?:\.\d+)?T?){2,})$", re.IGNORECASE),
]
TABLE_KEYWORD_PATTERNS = [
    "\u8d77\u91cd|\u5e45\u5ea6|\u81c2\u957f|\u8f7d\u8377|\u989d\u5b9a|\u500d\u7387|\u5de5\u51b5|\u6027\u80fd\u53c2\u6570",
    r"Lifting|capacity|Radius|Boom|Jib|load chart|hoisting height",
    "Traglasten|Ausladung|Hubh[o\u00f6]he|Fl[a\u00e4]che",
]
CATALOG_KEYWORD_PATTERN = re.compile(
    "\u6280\u672f\u53c2\u6570|\u53c2\u6570\u8868|\u4ea7\u54c1\u76ee\u5f55|\u578b\u8c31|\u9009\u578b|catalog|series|\u5854\u540a|\u5854\u673a|\u52a8\u81c2\u5854\u540a|\u5e73\u5934\u5854\u540a",
    re.IGNORECASE,
)
INDEX_KEYWORD_PATTERN = re.compile(
    "\u76ee\u5f55|\u7d22\u5f15|contents|index|\u4ea7\u54c1\u5217\u8868|\u7cfb\u5217|\u673a\u578b\u603b\u89c8|\u4ea7\u54c1\u603b\u89c8|\u4ea7\u54c1\u7b80\u4ecb",
    re.IGNORECASE,
)


def _looks_like_index_page(page_text: str) -> bool:
    txt = normalize_text(page_text)
    if not txt:
        return False
    if INDEX_KEYWORD_PATTERN.search(txt):
        return True
    page_models = extract_model_candidates(txt)
    return len(page_models) >= 5 and "杞借嵎" not in txt and "capacity" not in txt.lower()


def _looks_like_cover_page(page_text: str, document_title: str) -> bool:
    txt = normalize_text(page_text)
    if not txt:
        return True
    title_core = _core_model(document_title)
    has_title_signal = bool(title_core and title_core in _core_model(txt))
    short_lines = [line.strip() for line in txt.splitlines() if line.strip()]
    dense_model_hits = len(extract_model_candidates(txt))
    return has_title_signal or (len(short_lines) <= 8 and dense_model_hits <= 1)


def _page_content_role(page_text: str, document_title: str) -> str:
    if _looks_like_index_page(page_text):
        return "index"
    if _looks_like_cover_page(page_text, document_title):
        return "cover"
    return "body"


def _build_page_model_hints(page_texts: list[str], document_title: str = "") -> dict[int, str]:
    hints: dict[int, str] = {}
    for idx, page_text in enumerate(page_texts, start=1):
        role = _page_content_role(page_text, document_title)
        if role != "body":
            continue
        page_models = extract_model_candidates(page_text)
        if page_models:
            hints[idx] = page_models[0]
    return hints


def _is_catalog_like_pdf(
    pdf_name: str,
    document_title: str,
    biz_type: str,
    route_doc_type: str,
    page_texts: list[str],
    model_names: list[str],
    page_model_hints: dict[int, str],
) -> bool:
    total_pages = len(page_texts)
    if total_pages < 10:
        return False

    all_models = list(dict.fromkeys([*model_names, *page_model_hints.values()]))
    combined = normalize_text(f"{pdf_name} {document_title}")
    has_catalog_signal = bool(CATALOG_KEYWORD_PATTERN.search(combined))
    cover_pages = sum(1 for txt in page_texts[:3] if _looks_like_cover_page(txt, document_title))
    index_pages = sum(1 for txt in page_texts[:5] if _looks_like_index_page(txt))
    body_pages = sum(1 for txt in page_texts if _page_content_role(txt, document_title) == "body")
    scan_only = not any(txt.strip() for txt in page_texts)

    if biz_type == "tower_crane" and len(all_models) >= 3:
        return True
    if biz_type == "tower_crane" and has_catalog_signal:
        return True
    if biz_type in {"tower_crane", "unknown"} and scan_only and has_catalog_signal:
        return True
    if route_doc_type in {"spec_sheet", "brochure"} and has_catalog_signal and len(all_models) >= 2:
        return True
    if len(all_models) >= 3 and (cover_pages >= 1 or index_pages >= 1) and body_pages >= max(3, total_pages // 2):
        return True
    return False


def _vlm_page_budget(total_pages: int, catalog_like: bool) -> int:
    base = max(1, getattr(settings, "max_vlm_pages_per_file", 6))
    if catalog_like:
        return min(total_pages, max(base * 8, 48))
    return min(total_pages, base)


def _table_signal_score(route_doc_type: str, route_difficulty: str, page_text: str) -> int:
    txt = normalize_text(page_text)
    if _looks_like_index_page(txt):
        return 0

    digits = len(re.findall(r"\d", txt))
    numeric_density = digits / max(len(txt), 1)
    numeric_lines = sum(1 for line in txt.splitlines() if len(re.findall(r"\d+(?:\.\d+)?", line)) >= 2)
    keyword_hits = sum(1 for pat in TABLE_KEYWORD_PATTERNS if re.search(pat, txt, re.IGNORECASE))
    unit_hits = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:m|t)\b|m\*\d|t\)", txt, re.IGNORECASE))

    score = keyword_hits * 3 + min(numeric_lines, 8) + min(unit_hits, 5)
    if numeric_density > 0.10:
        score += 2
    if numeric_density > 0.18:
        score += 2
    if route_doc_type == "spec_sheet":
        score += 3
    elif route_doc_type == "manual":
        score += 1
    if route_difficulty == "text_rich":
        score += 1
    return score


def _select_table_pages(
    route_doc_type: str,
    route_difficulty: str,
    page_texts: list[str],
    page_model_hints: dict[int, str] | None = None,
    catalog_like: bool = False,
    document_title: str = "",
) -> list[int]:
    total_pages = len(page_texts)
    budget = _vlm_page_budget(total_pages, catalog_like)

    if catalog_like:
        body_pages = [idx for idx, txt in enumerate(page_texts, start=1) if _page_content_role(txt, document_title) == "body"]
        if page_model_hints:
            hinted_pages = [idx for idx in sorted(page_model_hints) if idx in body_pages]
            if len(hinted_pages) >= 3:
                return hinted_pages[:budget]
        if total_pages <= budget:
            return body_pages or list(range(1, total_pages + 1))

    scored: list[tuple[int, int]] = []
    for idx, page_text in enumerate(page_texts, start=1):
        if _page_content_role(page_text, document_title) != "body":
            continue
        score = _table_signal_score(route_doc_type, route_difficulty, page_text)
        if score > 0:
            scored.append((idx, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    threshold = 7 if route_doc_type == "spec_sheet" else 8
    chosen = [idx for idx, score in scored if score >= threshold]
    return chosen[:budget]


def _select_table_pages_fallback(
    biz_type: str,
    route_doc_type: str,
    route_difficulty: str,
    page_texts: list[str],
    page_model_hints: dict[int, str] | None = None,
    catalog_like: bool = False,
    document_title: str = "",
) -> tuple[list[int], str]:
    total_pages = len(page_texts)
    if total_pages == 0:
        return [], "no_pages"

    max_pages = _vlm_page_budget(total_pages, catalog_like)
    nonempty_pages = [idx for idx, txt in enumerate(page_texts, start=1) if txt.strip()]
    scan_only_ratio = 1.0 - (len(nonempty_pages) / total_pages)

    if catalog_like:
        body_pages = [idx for idx, txt in enumerate(page_texts, start=1) if _page_content_role(txt, document_title) == "body"]
        if page_model_hints:
            hinted_pages = [idx for idx in sorted(page_model_hints) if idx in body_pages]
            if len(hinted_pages) >= 3:
                return hinted_pages[:max_pages], "catalog_model_pages"
        if total_pages <= max_pages:
            return body_pages or list(range(1, total_pages + 1)), "catalog_full_scan"

    if route_difficulty == "scan_rich" and (scan_only_ratio >= 0.8 or not nonempty_pages):
        return list(range(1, min(total_pages, max_pages) + 1)), "scan_rich_image_pdf"

    if biz_type != "unknown" and total_pages <= max(max_pages, 8):
        return list(range(1, min(total_pages, max_pages) + 1)), "small_known_biz_pdf"

    scored = [
        (idx, _table_signal_score(route_doc_type, route_difficulty, page_text))
        for idx, page_text in enumerate(page_texts, start=1)
        if _page_content_role(page_text, document_title) == "body"
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    top_scored = [idx for idx, score in scored[:max_pages] if score > 0]
    if top_scored:
        return top_scored, "top_scored_pages"

    if total_pages <= 3:
        return list(range(1, total_pages + 1)), "small_pdf_last_resort"

    return [], "no_fallback"


def _safe_move(src: Path, dst_dir: Path) -> bool:
    target_path = _available_target_path(dst_dir, src.name, src)
    for _ in range(5):
        try:
            shutil.move(str(src), str(target_path))
            return True
        except PermissionError:
            time.sleep(2)
    return False


async def extract_document_title(md_text: str, pdf_path: Path, vlm_engine: Optional[EngineVLM] = None) -> str:
    cover_text = md_text[:1500]

    h1_matches = re.findall(r"^\s*#\s+(.+)$", cover_text, re.MULTILINE)
    for h1 in h1_matches:
        cleaned_h1 = re.sub(r"[*`_]", "", h1).strip()
        if len(cleaned_h1) > 4 and not TITLE_SKIP_RE.search(cleaned_h1):
            return cleaned_h1

    lines = [line.strip() for line in cover_text.split("\n") if len(line.strip()) > 3]
    for line in lines[:15]:
        if any(keyword.lower() in line.lower() for keyword in TITLE_KEYWORDS):
            clean_line = re.sub(r"[*#`>_]", "", line).strip()
            if 5 <= len(clean_line) <= 120:
                return clean_line

    if vlm_engine:
        prompt = (
            "Extract the complete official document title from the following OCR or Markdown text.\n"
            "Return only the title itself. If unsure, return UNKNOWN.\n\n"
            f"Text:\n{cover_text[:800]}"
        )
        try:
            vlm_title = await vlm_engine.chat_text(prompt)
            if vlm_title and "UNKNOWN" not in vlm_title.upper() and len(vlm_title) < 120:
                return vlm_title.strip()
        except Exception as exc:
            print(f"  Warning: title extraction via VLM failed: {exc}")

    clean_name = re.sub(r"\(\d+\)", "", pdf_path.stem).strip()
    return clean_name.replace("_", " ")


async def extract_multi_model_names(md_text: str, pdf_path: Path) -> list[str]:
    candidates: list[str] = []
    title = await extract_document_title(md_text, pdf_path)

    for source_text in [pdf_path.stem.replace("_", " "), title, md_text[:200000]]:
        for model in extract_model_candidates(source_text):
            if model not in candidates:
                candidates.append(model)

    if not candidates:
        cleaned_name = re.sub(r"\(\d+\)", "", pdf_path.stem).strip().replace("_", " ")
        for model in extract_model_candidates(cleaned_name):
            if model not in candidates:
                candidates.append(model)
        if not candidates and is_valid_device_model(cleaned_name):
            candidates.append(_normalize_model_token(cleaned_name))

    return list(dict.fromkeys(candidates))


def normalize_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    try:
        return float(text)
    except ValueError:
        return None


async def process_pdf(pdf_path: Path, vlm_engine: EngineVLM, loader: DmLoader, job_id: str) -> bool:
    import hashlib

    pdf_bytes = pdf_path.read_bytes()
    file_sha1 = generate_fingerprint(pdf_bytes)
    biz_type = DocumentClassifier.identify_biz_type(pdf_path.name)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    route_tags = DocumentClassifier.tag_3d(pdf_path.name, "")

    print(f"\nStart parsing: {pdf_path.name}")

    all_params: list[DeviceParam] = []
    all_images: list[FileLink] = []
    all_file_links: list[FileLink] = []
    all_main_devices: dict[int, DeviceMain] = {}
    extracted_pages_count = 0
    material_mapper = MaterialMapper()

    def build_device_id(model_name: str) -> int:
        normalized_model = _normalize_model_token(model_name)
        digest = hashlib.md5(f"{file_sha1}_{normalized_model}".encode("utf-8")).hexdigest()
        return int(digest, 16) % (10**8)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_texts = [normalize_text(doc[i].get_text("text")) for i in range(len(doc))]
        preview_text = "\n".join(page_texts[:5])
        route_tags = DocumentClassifier.tag_3d(pdf_path.name, preview_text)
        md_text = preview_text
        use_mineru_route = route_tags.difficulty == "scan_rich" or route_tags.doc_type == "brochure"
        mineru_asset_dir: Optional[Path] = None

        if use_mineru_route:
            print(f"  Route: MinerU OCR for {pdf_path.name}")
            mineru_client = MinerUClient(DEFAULT_MINERU_SERVICE_URL)
            try:
                result = mineru_client.convert_to_markdown_stream(
                    pdf_path,
                    end_pages=1000,
                    is_ocr=True,
                    formula_enable=True,
                    table_enable=True,
                    language="ch (Chinese, English, Chinese Traditional)",
                    backend="hybrid-auto-engine",
                    url="http://localhost:30000",
                )
                mineru_output_dir = IMG_DIR / "mineru_assets" / file_sha1
                materialized = mineru_client.materialize_result(result, pdf_path, mineru_output_dir)
                md_text = materialized.markdown_text or preview_text
                mineru_asset_dir = materialized.asset_dir
            except Exception as exc:
                print(f"  Warning: MinerU failed, fall back to per-page VLM. {exc}")
                use_mineru_route = False
                md_text = preview_text
                mineru_asset_dir = None

        if biz_type == "unknown":
            sniff = DocumentClassifier.identify_biz_type_from_text(md_text[:5000])
            if sniff != "unknown":
                biz_type = sniff

        document_title = await extract_document_title(md_text, pdf_path, vlm_engine)
        print(f"  Document title: {document_title}")

        raw_model_names = await extract_multi_model_names(md_text, pdf_path)
        # Merge full-document page-level model signals so catalog PDFs can register
        # all devices (e.g. QTZ + ZSL/ZSC pairs) before per-page extraction starts.
        for page_text in page_texts:
            for page_model in extract_model_candidates(page_text):
                if page_model not in raw_model_names:
                    raw_model_names.append(page_model)
        model_names = [model for model in raw_model_names if is_valid_device_model(model)]

        if not model_names:
            fallback_name = pdf_path.stem.upper()
            if is_valid_device_model(fallback_name):
                model_names = [fallback_name]
            else:
                print(f"  Warning: no valid model name found for {pdf_path.name}")
                return False

        known_models = list(dict.fromkeys(_normalize_model_token(model) for model in model_names))
        page_model_hints = _build_page_model_hints(page_texts, document_title)
        catalog_like = _is_catalog_like_pdf(
            pdf_path.name,
            document_title,
            biz_type,
            route_tags.doc_type,
            page_texts,
            known_models,
            page_model_hints,
        )

        async def ensure_device_record(model_name: str) -> int:
            normalized_model = _normalize_model_token(model_name)
            display_model = _display_model_token(normalized_model)
            if normalized_model not in known_models:
                known_models.append(normalized_model)
            dev_id = build_device_id(normalized_model)
            if dev_id in all_main_devices:
                return dev_id

            mapping_result = await material_mapper.map_equipment(document_title, normalized_model, vlm_engine, biz_type=biz_type)
            display_model = _display_model_token(normalized_model)
            all_main_devices[dev_id] = DeviceMain(
                id=dev_id,
                matnum=(mapping_result.get("matnum") or "").strip() or None,
                caterycode=(mapping_result.get("caterycode") or "").strip(),
                equipment_type=mapping_result.get("std_name", "UNKNOWN_DEVICE"),
                matname=display_model[:100],
                spec=display_model[:100],
                remarks=f"vlm_enriched | job_id:{job_id}",
                create_time=now,
                update_time=now,
            )
            return dev_id

        for model_name in model_names:
            print(f"  Locked model: {_display_model_token(model_name)}")

        pdf_object_name = f"pdf/{datetime.now().strftime('%Y%m%d')}/{pdf_path.name}"
        pdf_file_url = f"{settings.image_public_base_url.rstrip('/')}/{pdf_object_name}"
        pdf_extract_engine = "mineru+qwen3-vl" if use_mineru_route else "fitz+qwen3-vl"

        try:
            await EngineImage._upload_to_minio(pdf_path, pdf_object_name, bucket_name=settings.minio_bucket)
            print(f"  PDF uploaded to MinIO: {pdf_file_url}")
        except Exception as exc:
            print(f"  Warning: PDF upload failed: {exc}")
            pdf_file_url = str(pdf_path)

        model_page_ranges: dict[str, list[int]] = {model: [] for model in known_models}
        current_model = known_models[0]
        current_dev_id = await ensure_device_record(current_model)
        skip_page_vlm = False
        selected_pages: list[int] = []

        if use_mineru_route:
            if len(model_names) == 1:
                mineru_params = build_device_params(
                    current_dev_id,
                    md_text,
                    source_page=1,
                    extract_engine="mineru_md",
                    confidence=0.92,
                    now=now,
                )
                if mineru_params:
                    all_params.extend(mineru_params)
                    extracted_pages_count = 1
                    skip_page_vlm = True
                    print(f"  MinerU extracted {len(mineru_params)} rows")

                    if mineru_asset_dir is not None:
                        try:
                            mineru_files = await collect_mineru_file_links(
                                md_text,
                                model_name=current_model,
                                biz_type=all_main_devices[current_dev_id].equipment_type,
                                device_id=current_dev_id,
                                assets_dir=mineru_asset_dir,
                                confidence=0.92,
                                now=now,
                            )
                            all_images.extend(mineru_files)
                            if mineru_files:
                                print(f"  MinerU kept {len(mineru_files)} useful images")
                        except Exception as exc:
                            print(f"  Warning: MinerU asset collection failed: {exc}")
                else:
                    print("  MinerU returned no valid rows, continue with per-page VLM")
            else:
                print(
                    f"  Detected {len(model_names)} models in one PDF; keep MinerU for text recovery "
                    "but continue per-page VLM to preserve device attribution."
                )

        if not skip_page_vlm:
            selected_pages = _select_table_pages(
                route_tags.doc_type,
                route_tags.difficulty,
                page_texts,
                page_model_hints=page_model_hints,
                catalog_like=catalog_like,
                document_title=document_title,
            )
            if not selected_pages:
                selected_pages, fallback_reason = _select_table_pages_fallback(
                    biz_type,
                    route_tags.doc_type,
                    route_tags.difficulty,
                    page_texts,
                    page_model_hints=page_model_hints,
                    catalog_like=catalog_like,
                    document_title=document_title,
                )
                if selected_pages:
                    print(f"  Table-page fallback hit ({fallback_reason}): {selected_pages}")
            if selected_pages:
                print(f"  Table-page plan ({len(selected_pages)}/{len(doc)}): {selected_pages}")
            selected_page_set = set(selected_pages)

            for page_index, page in enumerate(doc):
                page_num = page_index + 1
                if selected_page_set and page_num not in selected_page_set:
                    continue

                page_text = page_texts[page_index]

                if biz_type == "unknown" and page_num <= 5:
                    sniff = DocumentClassifier.identify_biz_type_from_text(page_text)
                    if sniff != "unknown":
                        biz_type = sniff
                        print(f"  Page {page_num} sniffed biz type: {biz_type}")

                hinted_model = page_model_hints.get(page_num)
                detected_model = hinted_model or choose_best_model_candidate(page_text, known_models, current_model)
                page_targets = collect_page_model_targets(
                    page_text,
                    known_models,
                    current_model,
                    hinted_model=hinted_model,
                    vlm_model=None,
                )
                if not page_targets and detected_model:
                    page_targets = [detected_model]
                for target_model in page_targets:
                    model_page_ranges.setdefault(target_model, []).append(page_num)
                if detected_model and detected_model != current_model:
                    guard_mapped = await material_mapper.get_matnum(
                        detected_model,
                        document_title,
                        vlm_engine,
                        biz_type=biz_type,
                    )
                    guard_caterycode = guard_mapped.get("caterycode", "") if isinstance(guard_mapped, dict) else ""
                    if not guard_caterycode and detected_model in {"RESCUED", "UNKNOWN", "XCMG"}:
                        print(f"  Warning: reject dirty model switch '{detected_model}'")
                    elif not guard_caterycode and len(detected_model) <= 4:
                        print(f"  Warning: reject short model switch '{detected_model}'")
                    else:
                        current_model = detected_model
                        current_dev_id = await ensure_device_record(current_model)
                        print(f"  Switched current model to {current_model} on page {page_num}")

                prompt = (
                    "Extract crane chart rows from this page.\n"
                    "Return compact JSON only:\n"
                    '{"rows": [[boom_length, radius, load], ...], "device_model": "model"}\n'
                    'If there is no useful chart data, return {"rows": []}.'
                )

                print(f"  Page {page_num}/{len(doc)} extracting with VLM for {current_model}")
                try:
                    if biz_type == "tower_crane":
                        page_pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        vlm_response_text = await vlm_engine.chat_multimodal_bytes(prompt, page_pixmap.tobytes("png"))
                        if not vlm_response_text:
                            print(f"  Warning: page {page_num} returned empty tower-crane response")
                            continue
                        parsed = vlm_engine._clean_json(vlm_response_text)
                        model_from_vlm = parsed.get("device_model")
                        raw_rows = []
                        for row in parsed.get("rows", []):
                            try:
                                if isinstance(row, list) and len(row) >= 3:
                                    raw_rows.append(
                                        type(
                                            "Row",
                                            (),
                                            {
                                                "arm_length": row[0],
                                                "lifting_radius": row[1],
                                                "load": row[2],
                                                "lifting_height": None,
                                                "condition_name": None,
                                            },
                                        )()
                                    )
                            except Exception:
                                pass
                    else:
                        model_from_vlm, raw_rows = await vlm_engine.extract_table(page, settings.vlm_image_dpi)
                except Exception as exc:
                    print(f"  Warning: page {page_num} extraction crashed: {exc}")
                    continue

                if model_from_vlm and model_from_vlm != "null":
                    candidate_model = choose_best_model_candidate(model_from_vlm.strip(), known_models, current_model)
                    if candidate_model and _core_model(candidate_model) != _core_model(current_model):
                        current_model = candidate_model
                        current_dev_id = await ensure_device_record(current_model)
                        print(f"  VLM corrected model to {current_model}")

                if page_num <= 3 or len(raw_rows) > 0:
                    try:
                        page_seq = sum(1 for item in all_images if item.biz_id == current_dev_id and item.source_page_start == page_num) + 1
                        file_link_snapshot = await EngineImage.extract_and_save(
                            page=page,
                            dev_id=current_dev_id,
                            page_num=page_num,
                            save_dir=IMG_DIR,
                            pdf_name=_display_model_token(current_model),
                            biz_name=all_main_devices[current_dev_id].matname,
                            page_seq=page_seq,
                        )
                        file_link_snapshot.biz_type = all_main_devices[current_dev_id].equipment_type
                        all_images.append(file_link_snapshot)
                        print(f"  Saved snapshot for page {page_num}")
                    except Exception as exc:
                        print(f"  Warning: snapshot failed on page {page_num}: {exc}")

                if not raw_rows:
                    continue

                clean_rows, fail_reason = QualityGate.inspect(raw_rows)
                if fail_reason:
                    print(f"  Quality gate warning on page {page_num}: {fail_reason}")
                    if not clean_rows:
                        continue

                for row in clean_rows:
                    arm_length = normalize_number(row.arm_length)
                    lifting_radius = normalize_number(row.lifting_radius)
                    load = normalize_number(row.load)
                    lifting_height = normalize_number(row.lifting_height)
                    param_fp = generate_fingerprint(current_dev_id, arm_length, lifting_radius, load)
                    all_params.append(
                        DeviceParam(
                            device_id=current_dev_id,
                            param_fingerprint=param_fp,
                            arm_length=arm_length,
                            lifting_radius=lifting_radius,
                            load=load,
                            lifting_height=lifting_height,
                            source_page=page_num,
                            extract_engine="vlm",
                            confidence=0.88,
                            condition_name=row.condition_name,
                            create_time=now,
                            update_time=now,
                        )
                    )
                    extracted_pages_count += 1

        for model_name in model_names:
            dev_id = await ensure_device_record(model_name)
            device = all_main_devices[dev_id]
            pages_for_model = model_page_ranges.get(model_name, [])
            if pages_for_model:
                source_page_start = min(pages_for_model)
                source_page_end = max(pages_for_model)
            else:
                source_page_start = 1
                source_page_end = max(len(doc), 1)

            all_file_links.append(
                FileLink(
                    biz_id=dev_id,
                    biz_name=device.matname,
                    biz_type=device.equipment_type,
                    order_type="PDF_MANUAL",
                    file_name=document_title[:200],
                    file_url=pdf_file_url,
                    file_size=pdf_path.stat().st_size,
                    source_page_start=source_page_start,
                    source_page_end=source_page_end,
                    extract_engine=pdf_extract_engine,
                    extract_confidence=1.0,
                    create_time=now,
                    update_time=now,
                )
            )

        import gc

        gc.collect()

        db_success = False
    try:
        await asyncio.to_thread(loader.upsert_mat_matcode, list(all_main_devices.values()))
        if all_images:
            await asyncio.to_thread(loader.upsert_sys_base_file, all_images)
        if all_file_links:
            await asyncio.to_thread(loader.upsert_sys_base_file, all_file_links)
        if all_params:
            await asyncio.to_thread(loader.upsert_pa_crane_parameters, all_params)

        db_success = True
        print(
            f"  DB upsert finished | devices={len(all_main_devices)} params={len(all_params)} "
            f"images={len(all_images)} valid_pages={extracted_pages_count}"
        )
    except Exception as exc:
        err_text = str(exc)
        print(f"  DB upsert failed: {exc}")
        if any(keyword in err_text.lower() for keyword in ["timeout", "timed out", _zh(r"\u8d85\u65f6"), "connect", "connection"]):
            target_dir = OCR_PENDING_DIR if route_tags.difficulty == "scan_rich" else RETRY_DIR
        else:
            target_dir = DLQ_DIR
    else:
        target_dir = DONE_DIR

    move_success = _safe_move(pdf_path, target_dir)
    if not move_success:
        print(f"  Fatal warning: could not move file {pdf_path.name}")
    elif target_dir != DONE_DIR:
        print(f"  File moved to {target_dir.name}")

    return db_success and extracted_pages_count > 0


async def main():
    print(f"Scanning directory: {INCOMING_DIR.absolute()}")
    _cleanup_working_dir("startup")

    pdf_files = list(sorted(INCOMING_DIR.rglob("*.pdf")))
    if settings.max_pdfs > 0:
        pdf_files = pdf_files[: settings.max_pdfs]

    if not pdf_files:
        print("No PDF files found in the incoming directory.")
        return

    job_id = f"JOB_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    print("==================================================")
    print(f"Start ETL batch job | job_id={job_id}")
    print("==================================================\n")

    vlm_engine = EngineVLM(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        timeout=settings.vlm_timeout,
    )

    loader = DmLoader(
        host=settings.dm_host,
        port=settings.dm_port,
        user=settings.dm_user,
        password=settings.dm_password,
        schema=settings.dm_schema,
    )

    success_cnt = 0
    fail_cnt = 0

    loader.connect()
    loader.start_job(job_id, len(pdf_files))
    try:
        for pdf_path in pdf_files:
            try:
                ok = await process_pdf(pdf_path, vlm_engine, loader, job_id)
                if ok:
                    success_cnt += 1
                else:
                    fail_cnt += 1
            except Exception as exc:
                fail_cnt += 1
                print(f"\nFatal file error for {pdf_path.name}: {exc}")
                try:
                    if _safe_move(pdf_path, DLQ_DIR):
                        print(f"  Moved crashed file to DLQ: {DLQ_DIR / pdf_path.name}")
                    else:
                        print("  Warning: failed to move crashed file to DLQ")
                except Exception:
                    print("  Warning: failed to move crashed file to DLQ")
            finally:
                _cleanup_working_dir(f"after {pdf_path.name}")

        final_status = "FAILED" if fail_cnt == len(pdf_files) else "SUCCESS"
        loader.finish_job(job_id, success_cnt, fail_cnt, final_status)
    finally:
        try:
            if hasattr(loader, "conn") and loader.conn:
                loader.conn.commit()
                print("Final DB commit completed.")
                loader.conn.close()
                loader.conn = None
                print("DB connection closed.")
        except Exception as close_err:
            print(f"Error while closing DB connection: {close_err}")
        finally:
            _cleanup_working_dir("shutdown")


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())

