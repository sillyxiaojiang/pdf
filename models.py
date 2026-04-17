import hashlib
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import dmPython

from text_utils import format_number, to_db_safe_text


@dataclass
class ExtractRow:
    arm_length: Optional[float]
    lifting_radius: Optional[float]
    load: Optional[float]
    lifting_height: Optional[float]
    condition_name: Optional[str]


@dataclass
class DeviceMain:
    id: int
    matname: str
    spec: str
    equipment_type: str
    manufacturer: Optional[str] = None
    model_code: Optional[str] = None
    matnum: Optional[str] = None
    caterycode: Optional[str] = None
    rated_power: Optional[str] = None
    production_capacity: Optional[str] = None
    remarks: str = ""
    create_time: Optional[str] = None
    update_time: Optional[str] = None


@dataclass
class DeviceParam:
    device_id: int
    param_fingerprint: str
    arm_length: Optional[float]
    lifting_radius: Optional[float]
    load: Optional[float]
    lifting_height: Optional[float]
    source_page: int
    extract_engine: str
    confidence: float
    condition_name: Optional[str]
    create_time: Optional[str] = None
    update_time: Optional[str] = None


@dataclass
class FileLink:
    biz_id: int
    biz_type: str
    biz_name: str
    order_type: str
    file_name: str
    file_url: str
    file_size: int
    source_page_start: int
    source_page_end: int
    extract_engine: str
    extract_confidence: float
    create_time: Optional[str] = None
    update_time: Optional[str] = None


def generate_fingerprint(*args) -> str:
    base = "_".join(str(a) for a in args).encode("utf-8")
    return hashlib.sha1(base).hexdigest()


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DmLoader:
    _HISTORICAL_MATNUM_REGEX = re.compile(r"^000\d{8}$")

    def __init__(self, host, port, user, password, schema):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.schema = schema
        self.conn = None
        self._matnum_lock = threading.Lock()
        self._existing_historical_matnums: set[str] | None = None
        self._next_historical_matnum_int: int | None = None

    def connect(self):
        if self.conn is not None:
            return
        try:
            self.conn = dmPython.connect(
                user=self.user,
                password=self.password,
                server=self.host,
                port=self.port,
            )
            with self.conn.cursor() as cur:
                cur.execute(f"SET SCHEMA {self.schema}")
            print("  [DB] connected successfully", flush=True)
        except Exception as exc:
            self.conn = None
            raise ConnectionError(f"DB connection failed: {exc}")

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            finally:
                self.conn = None

    def _ensure_conn(self):
        if not self.conn:
            raise RuntimeError("Database is not connected, call connect() first")

    def _execute_many_with_fallback(self, sql_template: str, data_tuples: list):
        if not data_tuples:
            return

        try:
            with self.conn.cursor() as cur:
                cur.executemany(sql_template, data_tuples)
        except Exception as exc:
            error_msg = str(exc).lower()
            if "parameter" in error_msg or "invalid" in error_msg or "mismatch" in error_msg:
                parts = sql_template.split("?")
                fallback_sql = "".join(
                    [parts[i] + f":{i + 1}" for i in range(len(parts) - 1)] + [parts[-1]]
                )
                with self.conn.cursor() as cur:
                    cur.executemany(fallback_sql, data_tuples)
            else:
                raise

    def _get_existing_columns(self, table_name: str) -> set[str]:
        self._ensure_conn()
        sql = "SELECT COLUMN_NAME FROM USER_TAB_COLUMNS WHERE TABLE_NAME = ?"
        with self.conn.cursor() as cur:
            cur.execute(sql, (table_name.upper(),))
            rows = cur.fetchall()
        return {str(row[0]).upper() for row in rows}

    @classmethod
    def _normalize_matnum(cls, value) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _is_historical_matnum(cls, value) -> bool:
        return bool(cls._HISTORICAL_MATNUM_REGEX.fullmatch(cls._normalize_matnum(value)))

    @classmethod
    def _needs_matnum_repair(cls, matnum, caterycode) -> bool:
        normalized = cls._normalize_matnum(matnum)
        normalized_caterycode = cls._normalize_matnum(caterycode)
        if not normalized:
            return True
        if normalized in {"DEFAULT_0000", "00000000000"}:
            return True
        if normalized_caterycode and normalized == normalized_caterycode:
            return True
        return not cls._is_historical_matnum(normalized)

    def _load_historical_matnum_pool(self):
        self._ensure_conn()
        if self._existing_historical_matnums is not None and self._next_historical_matnum_int is not None:
            return

        sql = """
            SELECT TRIM(MATNUM)
            FROM MAT_MATCODE
            WHERE MATNUM IS NOT NULL
              AND TRIM(MATNUM) <> ''
              AND REGEXP_LIKE(TRIM(MATNUM), '^000[0-9]{8}$')
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        matnums = {self._normalize_matnum(row[0]) for row in rows if self._normalize_matnum(row[0])}
        max_value = max((int(value) for value in matnums), default=0)
        self._existing_historical_matnums = matnums
        self._next_historical_matnum_int = max_value + 1

    def _reserve_next_historical_matnum(self) -> str:
        self._load_historical_matnum_pool()
        assert self._existing_historical_matnums is not None
        assert self._next_historical_matnum_int is not None

        while True:
            candidate = f"{self._next_historical_matnum_int:011d}"
            self._next_historical_matnum_int += 1
            if candidate not in self._existing_historical_matnums:
                self._existing_historical_matnums.add(candidate)
                return candidate

    def _assign_historical_matnums(self, items: List[DeviceMain]):
        if not items:
            return

        with self._matnum_lock:
            self._load_historical_matnum_pool()
            assigned_in_batch: set[str] = set()

            for item in items:
                current_matnum = self._normalize_matnum(item.matnum)
                if self._needs_matnum_repair(current_matnum, item.caterycode) or current_matnum in assigned_in_batch:
                    item.matnum = self._reserve_next_historical_matnum()
                else:
                    item.matnum = current_matnum
                    if self._existing_historical_matnums is not None:
                        self._existing_historical_matnums.add(current_matnum)
                assigned_in_batch.add(self._normalize_matnum(item.matnum))

    def start_job(self, job_id: str, total_files: int):
        self._ensure_conn()
        sql = "INSERT INTO ETL_JOB_RUN (JOB_ID, STATUS, TOTAL_FILES) VALUES (?, 'RUNNING', ?)"
        with self.conn.cursor() as cur:
            cur.execute(sql, (job_id, total_files))
        self.conn.commit()

    def finish_job(self, job_id: str, success_count: int, failed_count: int, status: str = "SUCCESS"):
        self._ensure_conn()
        sql = (
            "UPDATE ETL_JOB_RUN "
            "SET END_TIME = CURRENT_TIMESTAMP, STATUS = ?, SUCCESS_FILES = ?, FAILED_FILES = ? "
            "WHERE JOB_ID = ?"
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, (status, success_count, failed_count, job_id))
        self.conn.commit()

    def upsert_mat_matcode(self, items: List[DeviceMain], commit: bool = True):
        self._ensure_conn()
        if not items:
            return

        deduped_items = {item.id: item for item in items}
        final_batch = list(deduped_items.values())
        self._assign_historical_matnums(final_batch)

        existing_cols = self._get_existing_columns("MAT_MATCODE")
        if "ID" not in existing_cols:
            raise Exception("MAT_MATCODE missing primary key column ID")

        optional_col_map = [
            ("CATERYCODE", "caterycode"),
            ("MATNAME", "matname"),
            ("SPEC", "spec"),
            ("EQUIPMENT_TYPE", "equipment_type"),
            ("MANUFACTURER", "manufacturer"),
            ("MODEL_CODE", "model_code"),
            ("MATNUM", "matnum"),
            ("RATED_POWER", "rated_power"),
            ("PRODUCTION_CAPACITY", "production_capacity"),
            ("REMARKS", "remarks"),
            ("CREATE_TIME", "create_time"),
            ("UPDATE_TIME", "update_time"),
        ]
        active_cols = [item for item in optional_col_map if item[0] in existing_cols]
        if not active_cols:
            raise Exception("MAT_MATCODE has no writable columns except ID")

        source_aliases = ["? AS ID"] + [f"? AS {col}" for col, _ in active_cols]
        insert_cols = ["ID"] + [col for col, _ in active_cols]

        data_tuples = []
        for item in final_batch:
            row = [item.id]
            now_value = item.create_time or item.update_time or _now_str()
            for col, attr in active_cols:
                value = getattr(item, attr)
                if col == "CATERYCODE":
                    value = to_db_safe_text(value, max_len=50)
                elif col == "MATNAME":
                    value = to_db_safe_text(value, max_len=200, fallback="UNNAMED_EQUIPMENT")
                elif col == "SPEC":
                    value = to_db_safe_text(value, max_len=200, fallback="UNKNOWN_SPEC")
                elif col in {"EQUIPMENT_TYPE", "MANUFACTURER", "MODEL_CODE"}:
                    value = to_db_safe_text(value, max_len=100)
                elif col == "MATNUM":
                    value = to_db_safe_text(value, max_len=50, fallback="00000000000")
                elif col in {"RATED_POWER", "PRODUCTION_CAPACITY"}:
                    value = to_db_safe_text(value, max_len=100)
                elif col == "REMARKS":
                    value = to_db_safe_text(value, max_len=500)
                elif col in {"CREATE_TIME", "UPDATE_TIME"}:
                    value = value or now_value
                row.append(value)
            data_tuples.append(tuple(row))

        update_expr = {
            "CATERYCODE": "T.CATERYCODE = CASE WHEN T.CATERYCODE IS NULL OR TRIM(T.CATERYCODE) = '' THEN S.CATERYCODE ELSE T.CATERYCODE END",
            "MATNAME": "T.MATNAME = CASE WHEN T.MATNAME IS NULL OR TRIM(T.MATNAME) = '' THEN S.MATNAME ELSE T.MATNAME END",
            "SPEC": "T.SPEC = CASE WHEN T.SPEC IS NULL OR TRIM(T.SPEC) = '' THEN S.SPEC ELSE T.SPEC END",
            "EQUIPMENT_TYPE": "T.EQUIPMENT_TYPE = CASE WHEN S.EQUIPMENT_TYPE IS NULL OR TRIM(S.EQUIPMENT_TYPE) = '' THEN T.EQUIPMENT_TYPE ELSE S.EQUIPMENT_TYPE END",
            "MANUFACTURER": "T.MANUFACTURER = CASE WHEN T.MANUFACTURER IS NULL OR TRIM(T.MANUFACTURER) = '' THEN S.MANUFACTURER ELSE T.MANUFACTURER END",
            "MODEL_CODE": "T.MODEL_CODE = CASE WHEN T.MODEL_CODE IS NULL OR TRIM(T.MODEL_CODE) = '' THEN S.MODEL_CODE ELSE T.MODEL_CODE END",
            "MATNUM": (
                "T.MATNUM = CASE "
                "WHEN T.MATNUM IS NULL OR TRIM(T.MATNUM) = '' "
                "OR TRIM(T.MATNUM) IN ('DEFAULT_0000', '00000000000') "
                "OR (T.CATERYCODE IS NOT NULL AND TRIM(T.CATERYCODE) <> '' AND TRIM(T.MATNUM) = TRIM(T.CATERYCODE)) "
                "OR NOT REGEXP_LIKE(TRIM(T.MATNUM), '^000[0-9]{8}$') "
                "THEN S.MATNUM ELSE T.MATNUM END"
            ),
            "RATED_POWER": "T.RATED_POWER = CASE WHEN T.RATED_POWER IS NULL OR TRIM(T.RATED_POWER) = '' THEN S.RATED_POWER ELSE T.RATED_POWER END",
            "PRODUCTION_CAPACITY": "T.PRODUCTION_CAPACITY = CASE WHEN T.PRODUCTION_CAPACITY IS NULL OR TRIM(T.PRODUCTION_CAPACITY) = '' THEN S.PRODUCTION_CAPACITY ELSE T.PRODUCTION_CAPACITY END",
            "REMARKS": "T.REMARKS = CASE WHEN S.REMARKS IS NULL OR TRIM(S.REMARKS) = '' THEN T.REMARKS ELSE S.REMARKS END",
            "CREATE_TIME": "T.CREATE_TIME = CASE WHEN T.CREATE_TIME IS NULL THEN S.CREATE_TIME ELSE T.CREATE_TIME END",
            "UPDATE_TIME": "T.UPDATE_TIME = S.UPDATE_TIME",
        }
        update_sql = ",\n                    ".join(
            update_expr[col]
            for col, _ in active_cols
            if col in update_expr
        )
        insert_col_sql = ", ".join(insert_cols)
        insert_val_sql = ", ".join([f"S.{col}" for col in insert_cols])

        sql = f"""
            MERGE INTO MAT_MATCODE T
            USING (
                SELECT
                    {", ".join(source_aliases)}
                FROM DUAL
            ) S
            ON (T.ID = S.ID)
            WHEN MATCHED THEN
                UPDATE SET
                    {update_sql}
            WHEN NOT MATCHED THEN
                INSERT ({insert_col_sql})
                VALUES ({insert_val_sql})
        """

        try:
            self._execute_many_with_fallback(sql, data_tuples)
            if commit:
                self.conn.commit()
            active_col_names = [col for col, _ in active_cols]
            print(f"  [MAT_MATCODE] upsert ok ({len(final_batch)} rows) cols=ID+{active_col_names}", flush=True)
        except Exception as exc:
            self.conn.rollback()
            raise Exception(f"MAT_MATCODE upsert failed: {exc}")

    def upsert_pa_crane_parameters(self, items: List[DeviceParam], commit: bool = True):
        self._ensure_conn()
        if not items:
            return

        import time

        deduped_items = {item.param_fingerprint: item for item in items}
        final_batch = list(deduped_items.values())
        base_id = int(time.time() * 1000) * 10000

        data_tuples = []
        for i, item in enumerate(final_batch):
            fake_id = base_id + i
            now_value = item.create_time or item.update_time or _now_str()
            data_tuples.append(
                (
                    fake_id,
                    item.device_id,
                    item.param_fingerprint,
                    format_number(item.arm_length),
                    format_number(item.lifting_radius),
                    format_number(item.load),
                    format_number(item.lifting_height),
                    to_db_safe_text(item.condition_name, max_len=200),
                    item.create_time or now_value,
                    item.update_time or now_value,
                )
            )

        sql = """
            MERGE INTO PA_CRANE_PARAMETERS T
            USING (
                SELECT
                    ? AS ID, ? AS DEVICE_ID, ? AS PARAM_FINGERPRINT,
                    ? AS ARM_LENGTH, ? AS LIFTING_RADIUS, ? AS LOAD,
                    ? AS LIFTING_HEIGHT, ? AS WORKING_CONDITION_NAME,
                    ? AS CREATE_TIME, ? AS UPDATE_TIME
                FROM DUAL
            ) S
            ON (T.DEVICE_ID = S.DEVICE_ID AND T.PARAM_FINGERPRINT = S.PARAM_FINGERPRINT)
            WHEN MATCHED THEN
                UPDATE SET
                    T.ARM_LENGTH = S.ARM_LENGTH,
                    T.LIFTING_RADIUS = S.LIFTING_RADIUS,
                    T.LOAD = S.LOAD,
                    T.LIFTING_HEIGHT = CASE WHEN S.LIFTING_HEIGHT IS NULL OR TRIM(S.LIFTING_HEIGHT) = '' THEN T.LIFTING_HEIGHT ELSE S.LIFTING_HEIGHT END,
                    T.WORKING_CONDITION_NAME = CASE WHEN S.WORKING_CONDITION_NAME IS NULL OR TRIM(S.WORKING_CONDITION_NAME) = '' THEN T.WORKING_CONDITION_NAME ELSE S.WORKING_CONDITION_NAME END,
                    T.UPDATE_TIME = S.UPDATE_TIME
            WHEN NOT MATCHED THEN
                INSERT (
                    ID, DEVICE_ID, PARAM_FINGERPRINT, ARM_LENGTH, LIFTING_RADIUS, LOAD,
                    LIFTING_HEIGHT, WORKING_CONDITION_NAME, CREATE_TIME, UPDATE_TIME
                ) VALUES (
                    S.ID, S.DEVICE_ID, S.PARAM_FINGERPRINT, S.ARM_LENGTH, S.LIFTING_RADIUS, S.LOAD,
                    S.LIFTING_HEIGHT, S.WORKING_CONDITION_NAME, S.CREATE_TIME, S.UPDATE_TIME
                )
        """

        try:
            self._execute_many_with_fallback(sql, data_tuples)
            if commit:
                self.conn.commit()
            print(f"  [PA_CRANE_PARAMETERS] upsert ok ({len(final_batch)} rows)", flush=True)
        except Exception as exc:
            self.conn.rollback()
            raise Exception(f"PA_CRANE_PARAMETERS upsert failed (batch={len(final_batch)}): {exc}")

    def upsert_sys_base_file(self, items: List[FileLink], commit: bool = True):
        self._ensure_conn()
        if not items:
            return

        import time

        deduped_items = {f"{item.biz_id}_{item.source_page_start}_{item.source_page_end}_{item.order_type}": item for item in items}
        final_batch = list(deduped_items.values())
        base_id = int(time.time() * 1000) * 10000

        data_tuples = []
        for i, item in enumerate(final_batch):
            fake_id = base_id + i
            now_value = item.create_time or item.update_time or _now_str()
            data_tuples.append(
                (
                    fake_id,
                    item.biz_id,
                    to_db_safe_text(item.biz_type, max_len=100, fallback="equipment"),
                    to_db_safe_text(item.biz_name, max_len=200, fallback="UNKNOWN_BIZ"),
                    to_db_safe_text(item.order_type, max_len=100, fallback="attachment"),
                    to_db_safe_text(item.file_name, max_len=200, fallback="UNNAMED_FILE"),
                    to_db_safe_text(item.file_url, max_len=900),
                    item.file_size,
                    item.source_page_start,
                    item.source_page_end,
                    to_db_safe_text(item.extract_engine, max_len=100, fallback="unknown"),
                    item.extract_confidence,
                    item.create_time or now_value,
                    item.update_time or now_value,
                )
            )

        sql = """
            MERGE INTO SYS_BASE_FILE T
            USING (
                SELECT
                    ? AS ID,
                    ? AS BIZ_ID, ? AS BIZ_TYPE, ? AS BIZ_NAME, ? AS ORDER_TYPE,
                    ? AS FILE_NAME, ? AS FILE_URL, ? AS FILE_SIZE,
                    ? AS SOURCE_PAGE_START, ? AS SOURCE_PAGE_END,
                    ? AS EXTRACT_ENGINE, ? AS EXTRACT_CONFIDENCE,
                    ? AS CREATE_TIME, ? AS UPDATE_TIME
                FROM DUAL
            ) S
            ON (T.BIZ_ID = S.BIZ_ID AND T.FILE_NAME = S.FILE_NAME AND T.ORDER_TYPE = S.ORDER_TYPE)
            WHEN MATCHED THEN
                UPDATE SET
                    T.BIZ_TYPE = S.BIZ_TYPE,
                    T.BIZ_NAME = S.BIZ_NAME,
                    T.FILE_URL = S.FILE_URL,
                    T.FILE_SIZE = S.FILE_SIZE,
                    T.SOURCE_PAGE_START = S.SOURCE_PAGE_START,
                    T.SOURCE_PAGE_END = S.SOURCE_PAGE_END,
                    T.EXTRACT_ENGINE = S.EXTRACT_ENGINE,
                    T.EXTRACT_CONFIDENCE = S.EXTRACT_CONFIDENCE,
                    T.UPDATE_TIME = S.UPDATE_TIME
            WHEN NOT MATCHED THEN
                INSERT (
                    ID, BIZ_ID, BIZ_TYPE, BIZ_NAME, ORDER_TYPE, FILE_NAME,
                    FILE_URL, FILE_SIZE, SOURCE_PAGE_START, SOURCE_PAGE_END,
                    EXTRACT_ENGINE, EXTRACT_CONFIDENCE, CREATE_TIME, UPDATE_TIME
                ) VALUES (
                    S.ID, S.BIZ_ID, S.BIZ_TYPE, S.BIZ_NAME, S.ORDER_TYPE, S.FILE_NAME,
                    S.FILE_URL, S.FILE_SIZE, S.SOURCE_PAGE_START, S.SOURCE_PAGE_END,
                    S.EXTRACT_ENGINE, S.EXTRACT_CONFIDENCE, S.CREATE_TIME, S.UPDATE_TIME
                )
        """

        try:
            self._execute_many_with_fallback(sql, data_tuples)
            if commit:
                self.conn.commit()
            print(f"  [SYS_BASE_FILE] upsert ok ({len(final_batch)} rows)", flush=True)
        except Exception as exc:
            self.conn.rollback()
            raise Exception(f"SYS_BASE_FILE upsert failed: {exc}")
