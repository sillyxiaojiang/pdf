from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

from models import DmLoader, DeviceMain, DeviceParam, FileLink, generate_fingerprint
from mineru_image_filter import filter_image_paths
from settings import load_settings

try:
    from minio import Minio
except Exception:  # pragma: no cover
    Minio = None

settings = load_settings()

TABLE_ROW_RE = re.compile(r"^\|(.+?)\|$")
NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _get_minio_client():
    if Minio is None:
        raise RuntimeError("minio dependency is not installed")
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=bool(settings.minio_secure),
    )


def _build_object_name(file_name: str) -> str:
    date_path = datetime.now().strftime("%Y%m%d")
    prefix = settings.minio_public_path_prefix.strip().strip("/")
    return f"{prefix}/{date_path}/{file_name}" if prefix else f"{date_path}/{file_name}"


def _build_http_file_url(object_name: str) -> str:
    base_url = settings.image_public_base_url.strip().rstrip("/")
    return f"{base_url}/{object_name.strip().lstrip('/')}".replace(" ", "")


def compute_device_id(file_sha1: str, model_name: str) -> int:
    digest = hashlib.md5(f"{file_sha1}_{model_name}".encode("utf-8")).hexdigest()
    return int(digest, 16) % (10**8)


def extract_model_name(md_text: str, fallback_model_name: str = "") -> str:
    patterns = [
        r"([A-Z]{2,4}[-\s_]?(?:\d{2,5})(?:[A-Z])?(?:-\d+)?(?:E|F|V)?)",
        r"(XCA\d{2,4}[A-Z]?-?\d*)",
        r"(TC\d{3,5}[A-Z]?(?:-\d+)?)",
        r"(QTZ\d{2,4})",
    ]
    for pat in patterns:
        match = re.search(pat, md_text)
        if match:
            return match.group(1).strip()
    fallback = (fallback_model_name or "").strip()
    return fallback or "UNKNOWN"


def extract_image_urls(md_text: str) -> list[str]:
    candidates: list[str] = []
    img_md_re = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<href>[^)]+)\)")

    for match in img_md_re.finditer(md_text):
        alt = (match.group("alt") or "").strip()
        href = (match.group("href") or "").strip()

        if alt and re.search(r"(?:^|/)images?/[^\s\)]+\.(png|jpg|jpeg|webp|bmp|gif)$", alt, re.IGNORECASE):
            candidates.append(alt)
            continue

        if href and not href.startswith("data:image/") and re.search(
            r"\.(png|jpg|jpeg|webp|bmp|gif)(\?|$)",
            href,
            re.IGNORECASE,
        ):
            candidates.append(href)

    for url in re.findall(r"https?://[^\s\)]+", md_text):
        if re.search(r"\.(png|jpg|jpeg|webp|bmp|gif)(\?|$)", url, re.IGNORECASE):
            candidates.append(url)

    cleaned = []
    for item in candidates:
        item = item.strip()
        if not item or item.startswith("data:image/"):
            continue
        if len(item) < 8:
            continue
        if re.search(r"(?:^|/)images?/[^\s\)]+\.(png|jpg|jpeg|webp|bmp|gif)$", item, re.IGNORECASE) or re.search(
            r"\.(png|jpg|jpeg|webp|bmp|gif)(\?|$)",
            item,
            re.IGNORECASE,
        ):
            cleaned.append(item)

    return list(dict.fromkeys(cleaned))


def extract_section_model(section_text: str, fallback_model_name: str = "") -> str:
    model = extract_model_name(section_text, fallback_model_name=fallback_model_name)
    if model and model != "UNKNOWN":
        return model
    return fallback_model_name or "UNKNOWN"


def split_markdown_sections(md_text: str) -> list[tuple[int | None, str]]:
    sections: list[tuple[int | None, str]] = []
    current_page: int | None = None
    buffer: list[str] = []
    page_marker_re = re.compile(r"^--\s*(\d+)\s*of\s*\d+\s*--$", re.IGNORECASE)

    for line in md_text.splitlines():
        marker = page_marker_re.match(line.strip())
        if marker:
            if buffer:
                sections.append((current_page, "\n".join(buffer).strip()))
                buffer = []
            current_page = int(marker.group(1))
            continue
        buffer.append(line)

    if buffer:
        sections.append((current_page, "\n".join(buffer).strip()))

    return sections


def _infer_page_from_url(raw_url: str) -> int | None:
    matches = re.findall(r"(?:p|page|page_)(\d+)", raw_url, re.IGNORECASE)
    if matches:
        try:
            return int(matches[-1])
        except Exception:
            return None
    return None


def _normalize_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.replace("\\n", " ").replace("|", " ").strip())


def _iter_markdown_rows(md_text: str) -> Iterable[list[str]]:
    for line in md_text.splitlines():
        line = line.strip()
        if not line or not TABLE_ROW_RE.match(line):
            continue
        if set(line.replace("|", "").strip()) <= {":", "-", " "}:
            continue
        cells = [_normalize_cell(c) for c in line.strip("|").split("|")]
        if len(cells) >= 3:
            yield cells


def extract_params(md_text: str) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []

    for cells in _iter_markdown_rows(md_text):
        nums = []
        for cell in cells:
            match = NUM_RE.search(cell)
            if match:
                nums.append(float(match.group(1)))
        if len(nums) >= 3:
            arm, radius, load = nums[0], nums[1], nums[2]
            if arm > 0 and radius > 0 and load > 0 and arm <= 2000 and radius <= 2000 and load <= 1000:
                rows.append((arm, radius, load))

    if rows:
        return list(dict.fromkeys(rows))

    for arm, radius, load in re.findall(
        r"(\d+(?:\.\d+)?)\s*[mM]?[\s,|]+\s*(\d+(?:\.\d+)?)\s*[mM]?[\s,|]+\s*(\d+(?:\.\d+)?)",
        md_text,
    ):
        triple = (float(arm), float(radius), float(load))
        if triple[0] > 0 and triple[1] > 0 and triple[2] > 0 and triple[0] <= 2000 and triple[1] <= 2000 and triple[2] <= 1000:
            rows.append(triple)

    return list(dict.fromkeys(rows))


def build_device_params(
    device_id: int,
    md_text: str,
    *,
    source_page: int = 1,
    extract_engine: str = "mineru_md",
    confidence: float = 0.9,
    now: str | None = None,
) -> list[DeviceParam]:
    current_time = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params: list[DeviceParam] = []

    for arm, radius, load in extract_params(md_text):
        fp = generate_fingerprint(device_id, arm, radius, load)
        params.append(
            DeviceParam(
                device_id=device_id,
                param_fingerprint=fp,
                arm_length=arm,
                lifting_radius=radius,
                load=load,
                lifting_height=None,
                source_page=source_page,
                extract_engine=extract_engine,
                confidence=confidence,
                condition_name=None,
                create_time=current_time,
                update_time=current_time,
            )
        )

    return params


async def collect_mineru_file_links(
    md_text: str,
    model_name: str,
    biz_type: str,
    device_id: int,
    assets_dir: Path | None,
    *,
    confidence: float = 0.9,
    now: str | None = None,
    existing_files: list[FileLink] | None = None,
) -> list[FileLink]:
    if assets_dir is None or not assets_dir.exists():
        return []

    image_urls = extract_image_urls(md_text)
    if not image_urls:
        return []

    try:
        client = _get_minio_client()
    except Exception as exc:
        print(f"  Warning: MinerU assets skipped because MinIO is unavailable: {exc}")
        return []

    sections = split_markdown_sections(md_text)
    candidate_meta: list[tuple[int, int, str, Path, str, str]] = []
    candidate_paths: list[Path] = []

    for idx, raw_url in enumerate(image_urls, start=1):
        raw_url = unquote(raw_url).strip()
        file_name = raw_url.rsplit("/", 1)[-1].split("?")[0]
        if file_name.startswith("images/"):
            file_name = file_name.split("images/", 1)[-1]
        if not re.search(r"\.(png|jpg|jpeg|webp|bmp|gif)$", file_name, re.IGNORECASE):
            continue

        source_img = (assets_dir / raw_url).resolve()
        if not source_img.exists():
            continue

        section_text = ""
        section_page = _infer_page_from_url(raw_url) or idx
        for page_num, content in sections:
            if page_num is not None and page_num == section_page:
                section_text = content
                break
        if not section_text and sections:
            section_text = sections[min(max(section_page - 1, 0), len(sections) - 1)][1]
        section_model = extract_section_model(section_text, fallback_model_name=model_name)

        candidate_paths.append(source_img)
        candidate_meta.append((section_page, idx, file_name, source_img, section_model, section_text))

    if not candidate_paths:
        return []

    kept_paths = await filter_image_paths(candidate_paths, context=model_name)
    kept_set = {p.resolve() for p in kept_paths}
    current_time = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    files: list[FileLink] = []
    existing_files = existing_files or []

    for page_num, image_seq, file_name, source_img, section_model, section_text in candidate_meta:
        if source_img.resolve() not in kept_set:
            continue

        object_name = _build_object_name(file_name)
        ext = source_img.suffix.lower()
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")

        try:
            client.fput_object(
                bucket_name=settings.minio_bucket,
                object_name=object_name,
                file_path=str(source_img),
                content_type=content_type,
            )
        except Exception as exc:
            print(f"  Warning: MinerU asset upload failed for {source_img.name}: {exc}")
            continue

        page_seq = sum(
            1
            for item in existing_files + files
            if item.biz_id == device_id and item.source_page_start == page_num and item.order_type == "workingCurve"
        ) + 1
        file_stem, file_ext = file_name.rsplit('.', 1) if '.' in file_name else (file_name, 'png')
        files.append(
            FileLink(
                biz_id=device_id,
                biz_type=biz_type,
                biz_name=section_model,
                order_type="workingCurve",
                file_name=f"{file_stem}_p{page_num}_{page_seq}.{file_ext}",
                file_url=_build_http_file_url(object_name),
                file_size=source_img.stat().st_size,
                source_page_start=page_num,
                source_page_end=page_num,
                extract_engine="mineru_md",
                extract_confidence=confidence,
                create_time=current_time,
                update_time=current_time,
            )
        )

    return files


def ingest_markdown(
    md_text: str,
    source_name: str = "document.md",
    assets_dir: Path | None = None,
    *,
    fallback_model_name: str = "",
    biz_type: str = "tower_crane",
    file_sha1: str | None = None,
    device_id: int | None = None,
    upload_assets: bool = True,
) -> tuple[DeviceMain, list[DeviceParam], list[FileLink]]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = extract_model_name(md_text, fallback_model_name=fallback_model_name)
    effective_sha1 = file_sha1 or generate_fingerprint(md_text.encode("utf-8"))
    current_dev_id = device_id or compute_device_id(effective_sha1, model_name)

    main_device = DeviceMain(
        id=current_dev_id,
        matname=model_name[:100],
        spec=model_name[:100],
        equipment_type=biz_type,
        remarks=f"mineru_md:{source_name} | sha1:{effective_sha1}",
        create_time=now,
        update_time=now,
    )

    params = build_device_params(
        current_dev_id,
        md_text,
        source_page=1,
        extract_engine="mineru_md",
        confidence=0.9,
        now=now,
    )

    files: list[FileLink] = []
    if upload_assets:
        files = asyncio.run(
            collect_mineru_file_links(
                md_text,
                model_name=model_name,
                biz_type=biz_type,
                device_id=current_dev_id,
                assets_dir=assets_dir,
                confidence=0.9,
                now=now,
            )
        )

    loader = DmLoader(
        host=settings.dm_host,
        port=settings.dm_port,
        user=settings.dm_user,
        password=settings.dm_password,
        schema=settings.dm_schema,
    )
    loader.connect()
    try:
        loader.upsert_mat_matcode([main_device])
        if params:
            loader.upsert_pa_crane_parameters(params)
        if files:
            loader.upsert_sys_base_file(files)
        print(f"Inserted MinerU markdown result: {source_name}")
        print(f"  - device_id: {current_dev_id}")
        print(f"  - params: {len(params)}")
        print(f"  - files: {len(files)}")
    finally:
        loader.close()

    return main_device, params, files


def main() -> None:
    sample_pdf = settings.incoming_dir / "100T-XCA100.pdf"
    sample_output_dir = sample_pdf.parent / f"{sample_pdf.stem}_mineru_test"
    sample_md = sample_output_dir / f"{sample_pdf.stem}.md"

    if not sample_md.exists():
        raise FileNotFoundError(f"Markdown file not found: {sample_md}")

    md_text = sample_md.read_text(encoding="utf-8", errors="ignore")
    ingest_markdown(
        md_text,
        source_name=sample_md.name,
        assets_dir=sample_output_dir,
        fallback_model_name=sample_pdf.stem,
    )


if __name__ == "__main__":
    main()
