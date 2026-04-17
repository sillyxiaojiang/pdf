from __future__ import annotations

from models import DmLoader
from settings import load_settings

settings = load_settings()


def init_equipment_dict() -> None:
    loader = DmLoader(
        host=settings.dm_host,
        port=settings.dm_port,
        user=settings.dm_user,
        password=settings.dm_password,
        schema=settings.dm_schema,
    )
    loader.connect()
    cursor = loader.conn.cursor()

    try:
        print("🚀 正在初始化标准物资设备字典库 (MDM)...")

        # 1. 扩容汽车起重机别名
        cursor.execute(
            """
            UPDATE PROGRAM.SYS_DICTIONARY
            SET NAME = '汽车起重机', CODE = '汽车吊,全地面,越野吊,STC,SAC,XCA,XCT,QY,LTM,ZAT,ZTC,ZRT'
            WHERE NAME = '汽车起重机'
            """
        )

        # 2. 扩容塔式起重机别名
        cursor.execute(
            """
            UPDATE PROGRAM.SYS_DICTIONARY
            SET NAME = '塔式起重机', CODE = '塔吊,平头塔吊,动臂塔吊,CJ,TC,TCT,QTZ,JTT,JTL'
            WHERE NAME = '塔式起重机'
            """
        )

        # 3. 新增直臂式随车吊（用 CODE 字段保存别名）
        cursor.execute(
            """
            MERGE INTO PROGRAM.SYS_DICTIONARY T
            USING (
                SELECT
                    NULL AS ID,
                    NULL AS PARENT_ID,
                    '00001496573' AS CODE,
                    '直臂式随车起重机' AS NAME,
                    NULL AS SORT,
                    NULL AS CREATE_USER,
                    CURRENT_TIMESTAMP AS CREATE_TIME,
                    NULL AS UPDATE_USER,
                    CURRENT_TIMESTAMP AS UPDATE_TIME,
                    0 AS IS_DELETED,
                    NULL AS CREATE_DEPT,
                    1 AS STATUS,
                    NULL AS TENANT_ID
                FROM DUAL
            ) S
            ON (T.CODE = S.CODE)
            WHEN MATCHED THEN
                UPDATE SET
                    T.NAME = S.NAME,
                    T.UPDATE_TIME = S.UPDATE_TIME,
                    T.STATUS = S.STATUS,
                    T.IS_DELETED = S.IS_DELETED
            WHEN NOT MATCHED THEN
                INSERT (
                    ID, PARENT_ID, CODE, NAME, SORT, CREATE_USER, CREATE_TIME,
                    UPDATE_USER, UPDATE_TIME, IS_DELETED, CREATE_DEPT, STATUS, TENANT_ID
                ) VALUES (
                    S.ID, S.PARENT_ID, S.CODE, S.NAME, S.SORT, S.CREATE_USER, S.CREATE_TIME,
                    S.UPDATE_USER, S.UPDATE_TIME, S.IS_DELETED, S.CREATE_DEPT, S.STATUS, S.TENANT_ID
                )
            """
        )

        loader.conn.commit()
        print("✅ 字典库别名 (ALIASES) 扩容完毕！")

    except Exception as e:
        loader.conn.rollback()
        print(f"❌ 初始化失败: {e}")
        raise
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        loader.close()


if __name__ == "__main__":
    init_equipment_dict()
