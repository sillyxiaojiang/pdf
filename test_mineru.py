from __future__ import annotations

import json
import zipfile
from pathlib import Path

from mineru_client import DEFAULT_MINERU_SERVICE_URL, MinerUClient


PDF_PATH = Path(r"C:\Users\Jcy\Desktop\企业级入库流程\测试文件\100T-XCA100.pdf")
OUTPUT_DIR = PDF_PATH.parent / f"{PDF_PATH.stem}_mineru_test"


def _save_markdown_and_assets(final_result, last_output):
    output_md = OUTPUT_DIR / f"{PDF_PATH.stem}.md"

    md_text = ""
    if isinstance(final_result, str):
        md_text = final_result
    elif isinstance(final_result, dict):
        md_text = json.dumps(final_result, ensure_ascii=False, indent=2)
    elif isinstance(final_result, list):
        for item in final_result:
            if isinstance(item, str) and item.strip().startswith("#"):
                md_text = item
                break
        if not md_text:
            md_text = json.dumps(final_result, ensure_ascii=False, indent=2)
    else:
        md_text = str(final_result)

    output_md.write_text(md_text, encoding="utf-8")

    zip_candidates = []
    for obj in [last_output, final_result]:
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, str) and item.lower().endswith(".zip"):
                    zip_candidates.append(Path(item))
        elif isinstance(obj, str) and obj.lower().endswith(".zip"):
            zip_candidates.append(Path(obj))

    for zip_path in zip_candidates:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(OUTPUT_DIR)
            print(f"📦 已解压 MinerU 结果包: {zip_path}")
            break

    print(f"✅ 解析完成，结果已保存到: {output_md}")
    print("\n===== 最终结果预览 =====\n")
    print(md_text[:5000])


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"找不到测试 PDF: {PDF_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🔎 测试文件: {PDF_PATH}")
    print(f"🌐 MinerU 服务: {DEFAULT_MINERU_SERVICE_URL}")
    print("🚀 开始执行 MinerU OCR 解析...")

    client = MinerUClient()
    result = client.convert_to_markdown_stream(
        PDF_PATH,
        end_pages=1000,
        is_ocr=True,
        formula_enable=True,
        table_enable=True,
        language="ch (Chinese, English, Chinese Traditional)",
        backend="hybrid-auto-engine",
        url="http://localhost:30000",
    )

    _save_markdown_and_assets(result.final_result, result.last_output)


if __name__ == "__main__":
    main()
