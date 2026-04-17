import re
import yaml
from pathlib import Path

def is_valid_device_model(model_name: str) -> bool:
    """设备型号基础清洗与过滤。"""
    if not model_name:
        return False

    clean_name = model_name.upper().strip()

    # 1. 长度过滤
    if len(clean_name) < 3 or len(clean_name) > 20:
        return False

    # 2. 字符白名单
    if not re.match(r"^[A-Z0-9\-\.\s]+$", clean_name):
        return False

    # 3. 行业禁用词黑名单
    kill_words = [
        "DIN", "ISO", "GB", "CE", "EN ",
        "CRANE", "TOWER", "TRUCK", "MACHINE",
        "GRUE", "GRU", "TOUR", "TORRE",
        "RESCUED", "UNKNOWN",
    ]
    for kw in kill_words:
        if kw in clean_name:
            return False

    # 4. 型号至少包含一个数字
    if not any(char.isdigit() for char in clean_name):
        return False

    return True


# ==========================================
# 企业级物资分类映射
# ==========================================
class MaterialMapper:
    _dict_cache = None

    @classmethod
    def _load_config(cls):
        """懒加载 YAML 配置文件。"""
        if cls._dict_cache is not None:
            return

        # 确保 equipment_aliases.yaml 与 mapper.py 在同一目录
        config_path = Path(__file__).parent / "equipment_aliases.yaml"
        if not config_path.exists():
            cls._dict_cache = []
            print("[Mapper] warning: equipment_aliases.yaml not found")
            return

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            cls._dict_cache = data.get("dictionaries", [])

    @classmethod
    async def map_equipment(cls, document_title: str, model_name: str, vlm_engine, biz_type: str = "") -> dict:
        cls._load_config()
        clean_model = model_name.upper().strip()

        def _pack_item(item: dict, match_mode: str) -> dict:
            return {
                "matnum": item.get("matnum", ""),
                "std_name": item.get("std_name", "未知设备"),
                "caterycode": item.get("category_code", ""),
                "match_mode": match_mode,
            }

        # 漏斗 1: 本地别名精确/前缀命中
        for item in cls._dict_cache:
            for alias in item.get("aliases", []):
                alias_upper = alias.upper()
                if clean_model == alias_upper or clean_model.startswith(alias_upper):
                    print(f"  [Mapper] alias hit: [{clean_model}] -> {item['std_name']} via [{alias_upper}]")
                    return _pack_item(item, "alias")

        # 漏斗 2: biz_type 分类兜底
        if biz_type and biz_type != "unknown":
            for item in cls._dict_cache:
                if item.get("biz_type") == biz_type:
                    print(f"  [Mapper] biz_type fallback: [{biz_type}] -> {item['std_name']}")
                    return _pack_item(item, "biz_type")

        # 漏斗 3: VLM 语义分类
        print(f"  [Mapper] VLM classify: {clean_model}")

        prompt = f"""
        你是一个工程物资分类专家。请判断下面设备型号属于哪一类工程机械。
        提取型号: {clean_model}
        文档标题: {document_title}
        当前业务分类: {biz_type}

        只输出标准设备名称，例如: 汽车起重机、固定塔式起重机、履带起重机、旋挖钻机。
        如果无法判断，只输出 UNKNOWN。
        """
        vlm_std_name = await vlm_engine.chat_text(prompt)

        if vlm_std_name and vlm_std_name != "UNKNOWN":
            for item in cls._dict_cache:
                if item["std_name"] == vlm_std_name:
                    return _pack_item(item, "vlm")

        return {"matnum": "", "std_name": "未知设备", "caterycode": "", "match_mode": "fallback"}


    @classmethod
    async def get_matnum(cls, model_name: str, document_title: str, vlm_engine, biz_type: str = "") -> dict:
        """
        Backward-compatible adapter used by main.py page-switch guard.
        Signature keeps model_name first to match existing call sites.
        """
        return await cls.map_equipment(document_title, model_name, vlm_engine, biz_type=biz_type)


# Backward-compatible alias
EquipmentMapper = MaterialMapper

