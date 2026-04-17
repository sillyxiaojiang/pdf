import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from models import ExtractRow


@dataclass
class RouteTags:
    biz_type: str
    doc_type: str
    difficulty: str


def _zh(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def _sanitize_lookup_text(text: str) -> str:
    clean_text = (text or "").upper()
    clean_text = clean_text.replace("_", " ")
    clean_text = clean_text.replace("/", "-")
    clean_text = re.sub(r"[\x00-\x1f]+", "-", clean_text)
    clean_text = re.sub(r"[()（）\[\]【】]+", "-", clean_text)
    clean_text = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Z0-9])", " ", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"(?<=[A-Z0-9])(?=[\u4e00-\u9fff])", " ", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"(?<=[A-Z0-9])\s+(?=[A-Z0-9])", "-", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    clean_text = re.sub(r"-{2,}", "-", clean_text)
    return clean_text


CRAWLER_PATTERNS = _compile_patterns(
    [
        r"\bSCC[-\s]?\d{2,6}[A-Z0-9.-]*\b",
        r"\bXGC[-\s]?\d{2,6}[A-Z0-9.-]*\b",
        r"\bQUY[-\s]?\d{2,6}[A-Z0-9.-]*\b",
        r"\bZCC[-\s]?\d{2,6}[A-Z0-9.-]*\b",
        r"\bCC[-\s]?\d+\b",
        r"\bLR[-\s]?\d{2,6}[A-Z0-9.-]*\b",
    ]
)
MOBILE_PATTERNS = _compile_patterns(
    [
        r"\bLTM[-\s]?\d{2,6}(?:[-.]\d+)*(?:[A-Z]\d*)?\b",
        r"\b(?:QY|QAY|SAC|SPS|XCT|XCA|STC|STB|SSC|SPC|ZTC|ZAT|ZCT|ZRT)[-\s]?\d{2,6}[A-Z0-9.-]*\b",
    ]
)
TOWER_PATTERNS = _compile_patterns(
    [
        r"\b(?:TCT|TCR|QTZ|QTD|QTP|EC-H)[-\s]?\d{2,6}[A-Z0-9.-]*\b",
        r"\bTC[-\s]?\d{2,6}[A-Z0-9.-]*\b",
        r"\b(?:ZSL|ZSC)[-\s]?\d{2,6}[A-Z0-9.-]*\b",
        r"\b(?:JT|JTL|JTT|JTZ)[-\w]*\d+\b",
        r"\b(?:STT|STL)[-\s]?\d{2,6}[A-Z0-9.-]*\b",
        r"\bST\d{2,5}[A-Z]?(?:-\d{2,5}[A-Z]?)?(?:-\d+(?:\.\d+)?[A-Z]?)?(?:-\d+(?:\.\d+)?[A-Z]?)?(?:-\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\b\d*HC-L\d*[A-Z0-9.-]*\b",
        r"\bSPT[-\s]?\d{2,5}[A-Z0-9.-]*\b",
        r"\bWQ\d{1,2}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\b(?:\d{1,2}[-\s]?)?CJ[-\s]?\d{2,4}(?:[-\s]?\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bC\d{4}[A-Z]?(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bD\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bF[0O][-\s]?\d{2}[A-Z](?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bFL\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bEL\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bH\d{1,2}(?:[-.]\d{2,3}[A-Z]?)\b",
        r"\bK\d{2}(?:[-.]\d{2,3}[A-Z]?)\b",
        r"\bM\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bMC\d{2,4}[A-Z]?(?:[-\s](?:H|K|L|TL|LH)\d{1,2})?\b",
        r"\bP\d{2,4}(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bQD\d{1,2}[A-Z]?(?:[-.]\d+(?:\.\d+)?[A-Z]?)?\b",
        r"\bR\d{2}(?:[-.]\d{2,3}[A-Z]?)\b",
        r"\bS\d{2,4}(?:[-\s](?:E\d|G\d|H\d{1,2}|K\d{1,2}|L\d{1,2}|LK\d{1,2}|LL\d{1,2}|LH\d{1,2}|TL\d{1,2}))\b",
    ]
)
CRAWLER_KEYWORDS = _compile_patterns(
    [
        _zh(r"\u5c65\u5e26\u8d77\u91cd\u673a"),
        _zh(r"\u5c65\u5e26\u540a"),
        r"crawler\s*crane",
    ]
)
MOBILE_KEYWORDS = _compile_patterns(
    [
        _zh(r"\u6c7d\u8f66\u8d77\u91cd\u673a"),
        _zh(r"\u5168\u5730\u9762\u8d77\u91cd\u673a"),
        _zh(r"\u8d8a\u91ce\u8f6e\u80ce\u8d77\u91cd\u673a"),
        _zh(r"\u968f\u8f66\u8d77\u91cd\u673a"),
        _zh(r"\u6298\u81c2"),
        _zh(r"\u76f4\u81c2"),
        r"truck\s*crane",
        r"all[-\s]?terrain",
        r"mobile\s*crane",
    ]
)
TOWER_KEYWORDS = _compile_patterns(
    [
        _zh(r"\u5854\u5f0f\u8d77\u91cd\u673a"),
        _zh(r"\u5854\u673a"),
        _zh(r"\u5854\u540a"),
        r"tower\s*crane",
    ]
)


class DocumentClassifier:
    """Three-dimensional document classifier."""

    _rules_cache: Optional[dict] = None

    @classmethod
    def _load_rules(cls) -> dict:
        if cls._rules_cache is not None:
            return cls._rules_cache

        rules_path = Path(__file__).with_name("rules.yaml")
        if not rules_path.exists():
            cls._rules_cache = {}
            return cls._rules_cache

        with rules_path.open("r", encoding="utf-8") as file:
            raw_rules = yaml.safe_load(file) or {}

        for doc_type, pattern in (raw_rules.get("doc_keywords", {}) or {}).items():
            if pattern:
                raw_rules.setdefault("_compiled_doc_keywords", {})[doc_type] = re.compile(str(pattern), re.IGNORECASE)

        cls._rules_cache = raw_rules
        return cls._rules_cache

    @staticmethod
    def _match_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    @classmethod
    def _identify_from_text(cls, text: str) -> str:
        raw_text = text or ""
        normalized_text = _sanitize_lookup_text(raw_text)

        if cls._match_any(normalized_text, CRAWLER_PATTERNS) or cls._match_any(raw_text, CRAWLER_KEYWORDS):
            return "crawler_crane"
        if cls._match_any(normalized_text, MOBILE_PATTERNS) or cls._match_any(raw_text, MOBILE_KEYWORDS):
            return "mobile_truck_crane"
        if cls._match_any(normalized_text, TOWER_PATTERNS) or cls._match_any(raw_text, TOWER_KEYWORDS):
            return "tower_crane"
        return "unknown"

    @classmethod
    def identify_biz_type(cls, file_name: str) -> str:
        return cls._identify_from_text(file_name or "")

    @classmethod
    def identify_biz_type_from_text(cls, text: str) -> str:
        return cls._identify_from_text(text or "")

    @classmethod
    def identify_doc_type(cls, text: str) -> str:
        rules = cls._load_rules()
        compiled = rules.get("_compiled_doc_keywords", {})
        for doc_type, pattern in compiled.items():
            if pattern and pattern.search(text or ""):
                return str(doc_type)
        return "unknown"

    @classmethod
    def identify_difficulty(cls, page_text: str) -> str:
        if len((page_text or "").strip()) >= 60:
            return "text_rich"
        return "scan_rich"

    @classmethod
    def tag_3d(cls, file_name: str, preview_text: str) -> RouteTags:
        return RouteTags(
            biz_type=cls.identify_biz_type(file_name),
            doc_type=cls.identify_doc_type(preview_text),
            difficulty=cls.identify_difficulty(preview_text),
        )


class QualityGate:
    """Basic extracted-row quality gate."""

    @staticmethod
    def _coerce_number(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    @classmethod
    def inspect(cls, rows: List[ExtractRow]) -> Tuple[List[ExtractRow], Optional[str]]:
        if not rows:
            return [], "NO_ROWS"

        clean_rows: List[ExtractRow] = []
        reject_count = 0

        for row in rows:
            arm_length = cls._coerce_number(row.arm_length)
            lifting_radius = cls._coerce_number(row.lifting_radius)
            load = cls._coerce_number(row.load)
            lifting_height = cls._coerce_number(row.lifting_height)

            if load is None or load <= 0:
                reject_count += 1
                continue
            if arm_length is None and lifting_radius is None:
                reject_count += 1
                continue
            if lifting_radius is not None and lifting_radius < 0:
                reject_count += 1
                continue

            clean_rows.append(
                ExtractRow(
                    arm_length=arm_length,
                    lifting_radius=lifting_radius,
                    load=load,
                    lifting_height=lifting_height,
                    condition_name=row.condition_name,
                )
            )

        if not clean_rows:
            return [], "ALL_ROWS_REJECTED"

        fail_reason = None
        if reject_count > 0:
            fail_reason = f"PARTIAL_REJECTED:{reject_count}"

        return clean_rows, fail_reason
