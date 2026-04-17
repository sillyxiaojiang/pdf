import re
import time
from pathlib import Path

from celery import Celery

from mineru_client import MinerUClient

# 初始化 Celery，连接 Redis
app = Celery(
    "ocr_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
)


@app.task(bind=True, max_retries=3)
def process_scan_rich_pdf(self, file_path_str: str, job_id: str):
    """
    独立 OCR 处理任务：专治高密度扫描件，全程在 CPU 运行，零显存占用。
    """
    pdf_path = Path(file_path_str)
    print(f"\n📦 [OCR Worker] 接收到复杂扫描件任务: {pdf_path.name} | Job: {job_id}")

    try:
        print("  ⏳ 正在调用 MinerU 流式接口进行深度版面分析与 OCR 识别...")

        client = MinerUClient()
        result = client.convert_to_markdown_stream(
            pdf_path,
            end_pages=1000,
            is_ocr=True,
            formula_enable=True,
            table_enable=True,
            language="ch (Chinese, English, Chinese Traditional)",
            backend="hybrid-auto-engine",
            url="http://localhost:30000",
        )

        final_result = result.final_result
        if isinstance(final_result, str):
            extracted_text = final_result
        else:
            extracted_text = str(final_result)

        safe_text = extracted_text.replace('"', '').replace("\n", " ").replace("\r", "")
        safe_text = re.sub(r"\s+", " ", safe_text).strip()

        output_txt_path = pdf_path.with_suffix(".txt")
        output_txt_path.write_text(safe_text, encoding="utf-8")

        print(f"  ✅ [OCR Worker] 处理完成！已生成纯文本文件: {output_txt_path.name}")
        return {"status": "SUCCESS", "original_file": pdf_path.name, "cleaned_text_path": str(output_txt_path)}
    except Exception as exc:
        print(f"  ❌ [OCR Worker] 任务崩溃: {exc}")
        raise self.retry(exc=exc, countdown=30)
