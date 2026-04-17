import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).replace("\x00", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def to_db_safe_text(text: Any, max_len: int | None = None, fallback: str = "") -> str:
    normalized = normalize_text(text)
    if not normalized:
        normalized = fallback

    deaccented = unicodedata.normalize("NFKD", normalized)
    deaccented = "".join(ch for ch in deaccented if not unicodedata.combining(ch))
    safe = deaccented.encode("gbk", "ignore").decode("gbk").strip()
    safe = re.sub(r"\s+", " ", safe)
    if not safe:
        safe = fallback
    if max_len is not None and max_len >= 0:
        safe = _truncate_gbk_bytes(safe, max_len)
    return safe


def _truncate_gbk_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0 or not text:
        return ""

    encoded = text.encode("gbk", "ignore")
    if len(encoded) <= max_bytes:
        return text

    kept_chars: list[str] = []
    used = 0
    for ch in text:
        chunk = ch.encode("gbk", "ignore")
        if not chunk:
            continue
        if used + len(chunk) > max_bytes:
            break
        kept_chars.append(ch)
        used += len(chunk)
    return "".join(kept_chars)


def normalize_model_name(text: Any) -> str:
    normalized = normalize_text(text).upper()
    normalized = normalized.replace("_", "-").replace("/", "-")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


def format_number(value: Any, scale: int = 4) -> str | None:
    if value is None or value == "":
        return None
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return to_db_safe_text(value, max_len=64) or None

    normalized = dec.quantize(Decimal(1)) if dec == dec.to_integral() else dec.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if scale >= 0 and "." in text:
        integer, fraction = text.split(".", 1)
        fraction = fraction[:scale].rstrip("0")
        text = integer if not fraction else f"{integer}.{fraction}"
    return text
