from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from settings import load_settings

settings = load_settings()


class MinerUImageFilter:
    """基于多模态模型的图片内容筛选器。"""

    _client: Optional[AsyncOpenAI] = None

    def __init__(self):
        if not settings.openai_api_key or not settings.openai_base_url or not settings.openai_model:
            raise RuntimeError("OPENAI/VLM 配置不完整，无法执行图片内容筛选")
        if MinerUImageFilter._client is None:
            MinerUImageFilter._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.vlm_timeout,
            )
        self.client = MinerUImageFilter._client
        self.model = settings.openai_model

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        ext = image_path.suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "gif": "image/gif",
            "bmp": "image/bmp",
        }.get(ext, "image/jpeg")
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        return f"data:{mime};base64,{b64}"

    async def should_keep(self, image_path: Path, context: str = "") -> tuple[bool, str]:
        data_url = self._image_to_data_url(image_path)
        prompt = (
            "你是起重机设备图片筛选器。请判断这张图片是否属于设备解析有价值的内容。"
            "仅当图片为工况图、载荷曲线、参数表、结构图、尺寸图、塔身图、吊臂图、配重关系图时返回 keep；"
            "如果是 logo、图标、控制面板、按钮、截图边角、吊钩单独示意、装饰图、无关照片，请返回 drop。"
            "输出严格 JSON，格式：{\"decision\":\"keep|drop\",\"reason\":\"...\"}。"
        )
        if context:
            prompt += f"\n上下文：{context}"

        rsp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        content = rsp.choices[0].message.content if rsp and rsp.choices else ""
        content = (content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
        import json
        import re

        try:
            start = content.find("{")
            end = content.rfind("}")
            obj = json.loads(content[start : end + 1]) if start != -1 and end != -1 else {}
        except Exception:
            m = re.search(r"\b(keep|drop)\b", content, re.IGNORECASE)
            decision = m.group(1).lower() if m else "drop"
            return decision == "keep", content[:200]

        decision = str(obj.get("decision", "drop")).lower()
        reason = str(obj.get("reason", ""))
        return decision == "keep", reason


async def filter_image_paths(image_paths: list[Path], context: str = "") -> list[Path]:
    """按图片内容筛选，保留 keep。"""
    if not image_paths:
        return []
    engine = MinerUImageFilter()
    kept: list[Path] = []
    for img in image_paths:
        try:
            ok, _ = await engine.should_keep(img, context=context)
            if ok:
                kept.append(img)
        except Exception:
            # 出错时保守保留，避免误删关键图片
            kept.append(img)
    return kept
