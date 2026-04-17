from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Centralized application settings."""

    base_dir: Path = Path(r"C:\Users\Jcy\Desktop\企业级入库流程")
    incoming_dir_name: str = "测试文件"
    done_dir_name: str = "lib_done"
    img_dir_name: str = "lib_working_curves"
    dlq_dir_name: str = "lib_dead_letter"

    static_file_base_url: str | None = "http://10.5.53.250:70"
    minio_parse_endpoint: str | None = Field(
        default="https://fabz-storage.dev.shecltd.com:9004",
        validation_alias="MINIO_CONFIG__PARSE_ENDPOINT",
    )
    minio_public_path_prefix: str = "paas/upload"

    minio_endpoint: str | None = Field(default="10.5.57.191:8092", validation_alias="MINIO_CONFIG__ENDPOINT")
    minio_access_key: str | None = Field(default="zVLHNeC30oBnsjqC", validation_alias="MINIO_CONFIG__ACCESS_KEY")
    minio_secret_key: str | None = Field(default="nJbDBep4fGgGglQPO5EGJjsJoeixiHLv", validation_alias="MINIO_CONFIG__SECRET_KEY")
    minio_secure: bool = Field(default=False, validation_alias="MINIO_CONFIG__SECURE")
    minio_bucket: str = Field(default="equipment", validation_alias="MINIO_CONFIG__BUCKET")

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None
    vlm_timeout: float = 90.0
    vlm_image_dpi: int = 120

    dm_host: str | None = "10.5.57.23"
    dm_port: int = 5236
    dm_user: str | None = "PROGRAM"
    dm_password: str | None = None
    dm_schema: str | None = "PROGRAM"

    max_pdfs: int = 0
    max_pdf_concurrency: int = 2
    max_page_concurrency: int = 4
    max_vlm_pages_per_file: int = 6
    ocr_pending_dir_name: str = "lib_ocr_pending"
    retry_dir_name: str = "lib_retry"

    model_config = SettingsConfigDict(
        env_file=(".env", "env.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def incoming_dir(self) -> Path:
        return self.base_dir / self.incoming_dir_name

    @property
    def done_dir(self) -> Path:
        return self.base_dir / self.done_dir_name

    @property
    def img_dir(self) -> Path:
        return self.base_dir / self.img_dir_name

    @property
    def dlq_dir(self) -> Path:
        return self.base_dir / self.dlq_dir_name

    @property
    def ocr_pending_dir(self) -> Path:
        return self.base_dir / self.ocr_pending_dir_name

    @property
    def retry_dir(self) -> Path:
        return self.base_dir / self.retry_dir_name

    @property
    def image_public_base_url(self) -> str:
        if self.minio_parse_endpoint:
            endpoint = self.minio_parse_endpoint.strip().rstrip("/")
            return f"{endpoint}/{self.minio_bucket}"

        base = (self.static_file_base_url or "").strip().rstrip("/")
        return f"{base}/{self.minio_bucket}" if self.minio_bucket else base


def load_settings() -> AppSettings:
    return AppSettings()


config = load_settings()
