import asyncio
import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import fitz
from openai import AsyncOpenAI

try:
    from minio import Minio  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    Minio = None

from models import ExtractRow, FileLink
from settings import load_settings

settings = load_settings()


class EngineVLM:
    """Vision-language extraction engine."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    async def chat_text(self, prompt: str, temperature: float = 0) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if response and response.choices:
            return (response.choices[0].message.content or "").strip()
        return ""

    async def chat_multimodal(self, prompt: str, image_path: Path, temperature: float = 0.1) -> str:
        b64_img = base64.b64encode(image_path.read_bytes()).decode()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(image_path.suffix.lower(), "image/png")
        data_url = f"data:{mime_type};base64,{b64_img}"

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=temperature,
        )
        if response and response.choices:
            return (response.choices[0].message.content or "").strip()
        return ""

    async def analyze_image(self, image_b64: str, prompt: str, temperature: float = 0) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
        )
        if response and response.choices:
            return (response.choices[0].message.content or "").strip()
        return ""

    def _clean_json(self, text: str) -> dict:
        content = (text or "").strip()
        if content:
            print(f"      [VLM preview] {content[:100]}...")

        if content.startswith("```"):
            lines = content.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            test_content = content
            if not test_content.endswith("}") and not test_content.endswith("]"):
                test_content += "]}"
            start = test_content.find("{")
            end = test_content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(test_content[start : end + 1])
        except Exception as exc:
            print(f"      [JSON parse warning] {exc}; trying regex salvage.")

        import re

        matches = re.findall(
            r"\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+|null|none)\s*\]",
            content,
            re.IGNORECASE,
        )
        if matches:
            rows = []
            for match in matches:
                try:
                    arm = float(match[0])
                    radius = float(match[1])
                    load = None if match[2].lower() in ["null", "none"] else float(match[2])
                    rows.append([arm, radius, load])
                except Exception:
                    pass
            print(f"      [JSON salvage] recovered {len(rows)} rows from partial output.")
            return {"rows": rows, "device_model": "RESCUED"}

        return {"rows": [], "device_model": None}

    async def extract_table(self, page: fitz.Page, dpi: int) -> Tuple[Optional[str], List[ExtractRow]]:
        requested_dpi = int(dpi or settings.vlm_image_dpi or 120)
        actual_dpi = max(72, min(requested_dpi, 200))
        image_b64 = base64.b64encode(
            page.get_pixmap(matrix=fitz.Matrix(actual_dpi / 72, actual_dpi / 72)).tobytes("jpeg")
        ).decode()

        prompt = (
            "Extract crane load-table data from this page and return compact JSON only.\n"
            "Interpret the table as [boom_length, radius, load].\n"
            'Use null when a single cell is unreadable.\n'
            'Return format: {"rows": [[9.2, 3.0, 33.4], [14.4, 3.0, 30.0]], "device_model": "LTM 1030"}'
        )

        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            ],
                        }
                    ],
                )
                if response and response.choices:
                    parsed = self._clean_json(response.choices[0].message.content)
                    rows: list[ExtractRow] = []
                    for row in parsed.get("rows", []):
                        try:
                            if isinstance(row, list) and len(row) >= 3:
                                extracted = ExtractRow(
                                    arm_length=float(row[0]) if row[0] is not None else None,
                                    lifting_radius=float(row[1]) if row[1] is not None else None,
                                    load=float(row[2]) if row[2] is not None else None,
                                    lifting_height=None,
                                    condition_name=None,
                                )
                                if extracted.arm_length or extracted.lifting_radius or extracted.load:
                                    rows.append(extracted)
                        except Exception:
                            pass
                    return parsed.get("device_model"), rows
            except Exception as exc:
                print(f"    [VLM retry {attempt + 1}] {type(exc).__name__}: {exc}")
                await asyncio.sleep(3)

        return None, []


class EngineImage:
    """Page snapshot extraction and MinIO upload helper."""

    _minio_client: Optional[Minio] = None

    @classmethod
    def _get_minio_client(cls):
        if Minio is None:
            raise RuntimeError("minio dependency is not installed")

        if cls._minio_client is None:
            if not settings.minio_endpoint:
                raise RuntimeError("MINIO_CONFIG__ENDPOINT is missing")
            if not settings.minio_access_key or not settings.minio_secret_key:
                raise RuntimeError("MINIO credentials are missing")

            cls._minio_client = Minio(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=bool(settings.minio_secure),
            )
        return cls._minio_client

    @staticmethod
    def _build_object_name(file_name: str, sub_dir: str = "") -> str:
        date_path = datetime.now().strftime("%Y%m%d")
        prefix = settings.minio_public_path_prefix.strip().strip("/")
        norm_sub = (sub_dir or "").strip().strip("/")
        parts = [part for part in [prefix, norm_sub, date_path, file_name] if part]
        return "/".join(parts)

    @staticmethod
    def _build_http_file_url(object_name: str) -> str:
        base_url = settings.image_public_base_url.strip().rstrip("/")
        clean_object_name = object_name.strip().lstrip("/")
        return f"{base_url}/{clean_object_name}"

    @staticmethod
    def _guess_content_type(file_path: Path) -> str:
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".pdf": "application/pdf",
        }.get(file_path.suffix.lower(), "application/octet-stream")

    @classmethod
    async def _upload_to_minio(cls, file_path: Path, object_name: str, bucket_name: Optional[str] = None):
        client = cls._get_minio_client()
        content_type = cls._guess_content_type(file_path)
        await asyncio.to_thread(
            client.fput_object,
            bucket_name=bucket_name or settings.minio_bucket,
            object_name=object_name,
            file_path=str(file_path),
            content_type=content_type,
        )

    @classmethod
    async def _upload_bytes_to_minio(cls, content: bytes, object_name: str, content_type: str):
        client = cls._get_minio_client()
        data = io.BytesIO(content)
        await asyncio.to_thread(
            client.put_object,
            bucket_name=settings.minio_bucket,
            object_name=object_name,
            data=data,
            length=len(content),
            content_type=content_type,
        )

    @classmethod
    async def extract_and_save(
        cls,
        page: fitz.Page,
        dev_id: int,
        page_num: int,
        save_dir: Path,
        pdf_name: str,
        biz_name: Optional[str] = None,
        page_seq: int = 1,
    ) -> FileLink:
        img_name = f"{dev_id}_curve_p{page_num}_{page_seq}.png"
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = pixmap.tobytes("png")

        object_name = cls._build_object_name(img_name, sub_dir="pdf")
        await cls._upload_bytes_to_minio(img_bytes, object_name, content_type="image/png")
        file_url = cls._build_http_file_url(object_name)
        file_size = len(img_bytes)

        try:
            img_path = save_dir / img_name
            if img_path.exists():
                img_path.unlink()
        except Exception:
            pass

        return FileLink(
            biz_id=dev_id,
            biz_type="equipment",
            biz_name=biz_name or pdf_name,
            order_type="workingCurve",
            file_name=img_name,
            file_url=file_url,
            file_size=file_size,
            source_page_start=page_num,
            source_page_end=page_num,
            extract_engine="fitz_render+minio",
            extract_confidence=1.0,
        )

        return FileLink(
            biz_id=dev_id,
            biz_type="equipment",
            biz_name=biz_name or pdf_name,
            order_type="workingCurve",
            file_name=img_name,
            file_url=file_url,
            file_size=img_path.stat().st_size,
            source_page_start=page_num,
            source_page_end=page_num,
            extract_engine="fitz_render+minio",
            extract_confidence=1.0,
        )
