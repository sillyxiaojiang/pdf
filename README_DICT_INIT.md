# 📚 物资主数据字典与别名配置指南 (MDM Configuration Guide)

## 1. 架构说明
本项目采用 **YAML 配置驱动** 的方式管理设备物资编码（MATNUM）与设备别名（Aliases）。
系统在解析 PDF 并提取出设备型号后，会通过三级漏斗（本地别名 -> 业务分类兜底 -> VLM 大模型语义）进行数据标准化，确保录入达梦数据库的主数据绝对纯净。

核心配置文件位于项目根目录：`equipment_aliases.yaml`

## 2. 如何新增或修改设备别名？
当发现新的未知设备型号（如 VLM 提取出了新的缩写 `QUY` 履带吊），**无需修改代码，无需修改数据库**，只需直接编辑 `equipment_aliases.yaml` 文件即可。

**操作步骤：**
1. 打开 `equipment_aliases.yaml`。
2. 找到对应的设备大类（例如“履带起重机”）。
3. 在 `aliases` 列表中追加新的别名前缀。

**配置示例：**
```yaml
  - matnum: "0000100104"
    std_name: "履带起重机"
    biz_type: "crawler_crane"
    aliases:
      - "QUY"   # <- 新增的别名
      - "XGC"
      - "SCC"
```

## 3. 生效机制
- **单次执行脚本 (`main.py`)**：保存 YAML 文件后，下次运行 `python main.py` 时立即生效。
- **Celery 异步队列 (分布式部署)**：由于 `MaterialMapper` 会将字典加载到类属性缓存 `_dict_cache` 中，如果在生产环境中修改了 YAML 文件，需要 **重启 Celery Worker** 以重新加载最新配置。

## 4. 字典设计铁律
- **优先级冲突**：系统按 YAML 中的顺序自上而下匹配。越具体的别名（如 `SPS`）应放在泛用别名之前。
- **大写统一**：虽然代码中做了 `.upper()` 兼容，但建议在 YAML 中统一使用大写字母维护英文前缀。
