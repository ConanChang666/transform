# -*- coding: utf-8 -*-
"""
主流程：DB 匯出 → 分段（中英分流）→ 翻譯（監聽）→ 匯回 DB

變更重點：
1) 將分段輸出分為 PARA_ZH_DIR / PARA_EN_DIR，減少混淆
2) 翻譯進度以「中文輸入檔數」為總量，且用三語完成數的最小值避免卡住
3) 以正則解析檔名並以「去語言碼」basename 對齊三語
4) DB 欄位使用 LONGTEXT；子行程用 sys.executable 啟動，並溫和終止
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import pymysql
import re
from typing import Tuple, Optional, Set

# --- 你自家的連線工具 ---
from db.connect_to_MySQL import MySQLConn

# --- 段落化的核心函數（英文路徑用） ---
from transform import read_transcript, reflow_to_paragraphs

# ======================
# 路徑與參數
# ======================
DB_NAME = "stock_market_data_lake"

INPUT_ZH_DIR = Path("./temp/input/zh")
INPUT_EN_DIR = Path("./temp/input/en")

PARA_ZH_DIR = Path("./temp/paragraphized/zh")
PARA_EN_DIR = Path("./temp/paragraphized/en")

ZHTW_OUTPUT_DIR = Path("./final/zhtw")
ZHCN_OUTPUT_DIR = Path("./final/zhcn")
EN_OUTPUT_DIR   = Path("./final/en")

for d in [
    INPUT_ZH_DIR, INPUT_EN_DIR,
    PARA_ZH_DIR, PARA_EN_DIR,
    ZHTW_OUTPUT_DIR, ZHCN_OUTPUT_DIR, EN_OUTPUT_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

# ======================
# 工具函數
# ======================

# 簡單判斷是否包含中文字符
def contains_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# 解析檔名：支援 *_paragraphized 與 .en/.zhtw/.zhcn 可有可無
NAME_RE = re.compile(
    r"""
    ^(?P<symbol>[^_]+)_
     (?P<year>\d{4})_
     (?P<quarter>\d+)
     (?:_paragraphized)?                # 可有可無
     (?:\.(?P<lang>en|zhtw|zhcn))?      # 可有可無
     \.txt$
    """,
    re.VERBOSE | re.IGNORECASE,
)

def parse_name_from_filename(filename: str) -> Tuple[str, str, str, str]:
    """
    給一個完整檔名（含副檔名），解析出 (symbol, year, quarter, lang)
    lang 若不存在，回傳空字串。
    """
    m = NAME_RE.match(filename)
    if not m:
        raise ValueError(f"無法解析檔名：{filename}")
    return (
        m.group("symbol"),
        m.group("year"),
        m.group("quarter"),
        (m.group("lang") or "").lower(),
    )

def base_no_lang_from_name(filename: str) -> str:
    """
    將 'AAPL_2023_4_paragraphized.en.txt' → 'AAPL_2023_4_paragraphized'
    """
    if filename.lower().endswith(".en.txt"):
        return filename[:-len(".en.txt")]
    if filename.lower().endswith(".zhtw.txt"):
        return filename[:-len(".zhtw.txt")]
    if filename.lower().endswith(".zhcn.txt"):
        return filename[:-len(".zhcn.txt")]
    if filename.lower().endswith(".txt"):
        # 沒語言碼的單純 .txt
        return filename[:-len(".txt")]
    raise ValueError(f"無法擷取 basename（去語言碼）：{filename}")

def partner_paths(base_no_lang: str) -> Tuple[Path, Path, Path]:
    """
    由「去語言碼」的 basename 組回三語檔案完整路徑
    """
    return (
        EN_OUTPUT_DIR / f"{base_no_lang}.en.txt",
        ZHTW_OUTPUT_DIR / f"{base_no_lang}.zhtw.txt",
        ZHCN_OUTPUT_DIR / f"{base_no_lang}.zhcn.txt",
    )

def collect_all_basenames_no_lang() -> Set[str]:
    """
    從三個輸出資料夾收集所有「去語言碼」的 basename，
    以 union 方式整合，確保就算某語言缺檔也能匯回。
    """
    basenames: Set[str] = set()
    for p in EN_OUTPUT_DIR.glob("*.en.txt"):
        basenames.add(base_no_lang_from_name(p.name))
    for p in ZHTW_OUTPUT_DIR.glob("*.zhtw.txt"):
        basenames.add(base_no_lang_from_name(p.name))
    for p in ZHCN_OUTPUT_DIR.glob("*.zhcn.txt"):
        basenames.add(base_no_lang_from_name(p.name))
    return basenames

# ======================
# 1) 從資料庫讀取並匯出成檔案（按語言分流）
# ======================
def export_from_db():
    print("--- 從資料庫讀取並匯出成檔案（中英分流） ---")
    with MySQLConn(DB_NAME) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT symbol, year, quarter, content FROM us_earnings_calls")
            records = cursor.fetchall()

    print(f"從資料庫讀取 {len(records)} 筆資料，按語言分流。")
    for record in records:
        symbol = record['symbol']
        year = str(record['year'])
        quarter = str(record['quarter'])
        file_stem = f"{symbol}_{year}_{quarter}.txt"
        content = record['content'] or ""

        if contains_chinese(content):
            file_path = INPUT_ZH_DIR / file_stem
            file_path.write_text(content, encoding="utf-8")
            print(f"匯出中文：{file_path}")
        else:
            file_path = INPUT_EN_DIR / file_stem
            file_path.write_text(content, encoding="utf-8")
            print(f"匯出英文：{file_path}")

# ======================
# 2) 執行段落化
# ======================
def run_paragraphizer():
    print("--- 開始執行段落化 ---")

    # 中文檔：走 transform 批次（通常會喂給 AI）
    print("處理中文文件（批次）...")
    zh_env = os.environ.copy()
    zh_env["TRANSFORM_INPUT_DIR"] = str(INPUT_ZH_DIR)
    zh_env["TRANSFORM_OUTPUT_DIR"] = str(PARA_ZH_DIR)
    zh_env["TRANSFORM_GLOB"] = "*.txt"
    zh_env["TRANSFORM_RESUME"] = "False"

    proc = subprocess.run(
        [sys.executable, "transform.py", "--batch"],
        env=zh_env,
        capture_output=True,
        text=True
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    # 英文檔：呼叫純 Python 函數
    print("處理英文文件（純 Python 函數）...")
    for file_path in INPUT_EN_DIR.glob("*.txt"):
        try:
            lines = read_transcript(str(file_path))
            paragraphized_text = reflow_to_paragraphs("\n".join(lines))
            out_path = PARA_EN_DIR / (file_path.stem + "_paragraphized.txt")
            out_path.write_text(paragraphized_text, encoding="utf-8")
            print(f"處理完成：{out_path}")
        except Exception as e:
            print(f"英文文件處理失敗 {file_path.name}: {e}")

    print("--- 段落化完成 ---")
    time.sleep(1)

# ======================
# 3) 執行翻譯（監聽進度）
# ======================
def run_translator():
    print("--- 開始執行翻譯程式（監聽模式） ---")

    # 只把「中文分段」交給 run_pipeline 翻譯
    process = subprocess.Popen(
        [
            sys.executable, "run_pipeline.py",
            "--para_dir", str(PARA_ZH_DIR),
            "--out_zhtw", str(ZHTW_OUTPUT_DIR),
            "--out_zhcn", str(ZHCN_OUTPUT_DIR),
            "--out_en",   str(EN_OUTPUT_DIR),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    total_files = len(list(PARA_ZH_DIR.glob("*.txt")))
    print(f"等待 {total_files} 個中文分段檔翻譯完成...")

    last_done = -1

    def done_count() -> int:
        return min(
            len(list(EN_OUTPUT_DIR.glob("*.en.txt"))),
            len(list(ZHTW_OUTPUT_DIR.glob("*.zhtw.txt"))),
            len(list(ZHCN_OUTPUT_DIR.glob("*.zhcn.txt"))),
        )

    try:
        while True:
            current = done_count()
            if current != last_done:
                print(f"已完成：{current}/{total_files}")
                last_done = current

            if current >= total_files:
                break

            # 也讀一下子行程輸出（避免緩衝塞住）
            if process.stdout and not process.stdout.closed:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        print(f"[translator] {line}")

            time.sleep(2)
    finally:
        # 溫和收斂
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            process.kill()
        # 列出最後錯誤輸出（若有）
        if process.stderr:
            err = process.stderr.read()
            if err:
                print(err)

    print("--- 翻譯程式執行完成 ---")

# ======================
# 4) 寫回資料庫
# ======================
def import_to_db():
    print("--- 將翻譯結果寫回資料庫 ---")

    basenames = collect_all_basenames_no_lang()
    print(f"預計寫回 {len(basenames)} 筆（以三語 union 為準）。")

    with MySQLConn(DB_NAME) as conn:
        with conn.cursor() as cursor:
            # 盡量用 LONGTEXT，避免內容被截斷
            try:
                cursor.execute("ALTER TABLE us_earnings_calls ADD COLUMN content_zhtw LONGTEXT")
                cursor.execute("ALTER TABLE us_earnings_calls ADD COLUMN content_zhcn LONGTEXT")
                cursor.execute("ALTER TABLE us_earnings_calls ADD COLUMN content_en LONGTEXT")
                conn.commit()
            except pymysql.err.OperationalError:
                print("資料庫欄位已存在，跳過新增。")
                conn.rollback()

            for base in sorted(basenames):
                try:
                    # 用一個「人工構造」的檔名來解析 symbol/year/quarter
                    # 這樣可以重用同一個 regex（lang 可無）
                    pseudo_name = f"{base}.txt"
                    symbol, year, quarter, _ = parse_name_from_filename(pseudo_name)

                    en_file, zhtw_file, zhcn_file = partner_paths(base)
                    content_en   = en_file.read_text(encoding="utf-8") if en_file.exists() else None
                    content_zhtw = zhtw_file.read_text(encoding="utf-8") if zhtw_file.exists() else None
                    content_zhcn = zhcn_file.read_text(encoding="utf-8") if zhcn_file.exists() else None

                    cursor.execute(
                        """
                        UPDATE us_earnings_calls
                        SET content_zhtw=%s, content_zhcn=%s, content_en=%s
                        WHERE symbol=%s AND year=%s AND quarter=%s
                        """,
                        (content_zhtw, content_zhcn, content_en, symbol, year, quarter)
                    )
                except Exception as e:
                    print(f"寫回失敗 {base}: {e}")

        conn.commit()
    print("--- 資料庫更新完成 ---")

# ======================
# 入口（Main）
# ======================
if __name__ == "__main__":
    # 重要提醒：請確保 run_pipeline.py 內為：
    # from opencc import OpenCC
    # cc = OpenCC('s2t')  # 不要帶 .json

    export_from_db()
    run_paragraphizer()
    run_translator()
    import_to_db()
    print("--- 所有流程已完成 ---")
