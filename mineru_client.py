from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from gradio_client import Client, handle_file


@dataclass
class MinerUConvertResult:
    final_result: Any
    last_output: Any | None = None


@dataclass
class MinerUMaterializedResult:
    markdown_text: str
    output_md_path: Path
    asset_dir: Path


DEFAULT_MINERU_SERVICE_URL = "http://10.5.68.19:7860"


class MinerUClient:
    """MinerU / PDF-Extract-Kit 的 Gradio 流式客户端封装。"""

    def __init__(self, service_url: str = DEFAULT_MINERU_SERVICE_URL, api_name: str = "/convert_to_markdown_stream"):
        self.service_url = service_url
        self.api_name = api_name
        self._client: Optional[Client] = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(self.service_url)
        return self._client

    def view_api(self):
        return self._get_client().view_api()

    @staticmethod
    def _extract_markdown_text(payload: Any) -> str:
        if isinstance(payload, str):
            return "" if payload.lower().endswith(".zip") else payload

        if isinstance(payload, dict):
            for key in ("markdown", "md", "content", "text"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return json.dumps(payload, ensure_ascii=False, indent=2)

        if isinstance(payload, (list, tuple)):
            text_candidates = [
                item
                for item in payload
                if isinstance(item, str) and item.strip() and not item.lower().endswith(".zip")
            ]
            if text_candidates:
                text_candidates.sort(
                    key=lambda item: (
                        item.strip().startswith("#"),
                        "| " in item or "\n|" in item,
                        len(item),
                    ),
                    reverse=True,
                )
                return text_candidates[0]
            return json.dumps(list(payload), ensure_ascii=False, indent=2)

        return str(payload)

    @staticmethod
    def _collect_zip_candidates(*payloads: Any) -> list[Path]:
        candidates: list[Path] = []

        def _walk(value: Any):
            if isinstance(value, (list, tuple)):
                for item in value:
                    _walk(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    _walk(item)
                return
            if isinstance(value, str) and value.lower().endswith(".zip"):
                candidates.append(Path(value))

        for payload in payloads:
            _walk(payload)

        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def materialize_result(
        self,
        result: MinerUConvertResult,
        pdf_path: str | Path,
        output_dir: str | Path,
    ) -> MinerUMaterializedResult:
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown_text = self._extract_markdown_text(result.final_result).strip()
        if not markdown_text:
            markdown_text = str(result.final_result)

        output_md_path = output_dir / f"{pdf_path.stem}.md"
        output_md_path.write_text(markdown_text, encoding="utf-8")

        for zip_path in self._collect_zip_candidates(result.last_output, result.final_result):
            if not zip_path.exists():
                continue
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_dir)

        return MinerUMaterializedResult(
            markdown_text=markdown_text,
            output_md_path=output_md_path,
            asset_dir=output_dir,
        )

    def convert_to_markdown_stream(
        self,
        pdf_path: str | Path,
        *,
        end_pages: int = 1000,
        is_ocr: bool = True,
        formula_enable: bool = True,
        table_enable: bool = True,
        language: str = "ch (Chinese, English, Chinese Traditional)",
        backend: str = "hybrid-auto-engine",
        url: str = "http://localhost:30000",
    ) -> MinerUConvertResult:
        """调用流式接口并返回最终结果。"""
        pdf_path = Path(pdf_path)
        job = self._get_client().submit(
            file_path=handle_file(str(pdf_path)),
            end_pages=end_pages,
            is_ocr=is_ocr,
            formula_enable=formula_enable,
            table_enable=table_enable,
            language=language,
            backend=backend,
            url=url,
            api_name=self.api_name,
        )

        last_output = None
        for output in job:
            last_output = output
            print(output)

        final_result = job.result()
        return MinerUConvertResult(final_result=final_result, last_output=last_output)
