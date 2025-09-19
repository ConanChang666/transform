import os
import subprocess
import time
from pathlib import Path
import pymysql
import re

# 導入你寫的連線程式
from db.connect_to_MySQL import MySQLConn

# 導入段落化程式中的核心函數
from transform import read_transcript, reflow_to_paragraphs

# --- 參數設定 ---
DB_NAME = "stock_market_data_lake"
# 為中英文資料建立不同的輸入目錄
INPUT_ZH_DIR = Path("./temp/input/zh")
INPUT_EN_DIR = Path("./temp/input/en")
PARA_OUTPUT_DIR = Path("./temp/paragraphized")
ZHTW_OUTPUT_DIR = Path("./final/zhtw")
ZHCN_OUTPUT_DIR = Path("./final/zhcn")
EN_OUTPUT_DIR = Path("./final/en")

# 確保所有目錄都存在
for d in [INPUT_ZH_DIR, INPUT_EN_DIR, PARA_OUTPUT_DIR, ZHTW_OUTPUT_DIR, ZHCN_OUTPUT_DIR, EN_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 簡單判斷是否包含中文字符
def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# --- 1. 從資料庫讀取並匯出成檔案（按語言分流） ---
def export_from_db():
    print("--- 從資料庫讀取並匯出成檔案 ---")
    with MySQLConn(DB_NAME) as conn:
        with conn.cursor() as cursor:
            # 這裡的資料表名稱請替換為你的實際名稱
            cursor.execute("SELECT symbol, year, quarter, content FROM us_earnings_calls")
            records = cursor.fetchall()

    print(f"從資料庫讀取 {len(records)} 筆資料，按語言分流。")
    for record in records:
        file_name = f"{record['symbol']}_{record['year']}_{record['quarter']}.txt"
        content = record['content']
        
        if contains_chinese(content):
            file_path = INPUT_ZH_DIR / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"匯出中文：{file_path}")
        else:
            file_path = INPUT_EN_DIR / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"匯出英文：{file_path}")

# --- 2. 執行段落化程式 ---
def run_paragraphizer():
    print("--- 開始執行段落化程式 ---")

    # 處理中文文件 (使用 AI 模型)
    print("處理中文文件（使用 AI 模型）...")
    env = os.environ.copy()
    env["TRANSFORM_INPUT_DIR"] = str(INPUT_ZH_DIR)
    env["TRANSFORM_OUTPUT_DIR"] = str(PARA_OUTPUT_DIR)
    env["TRANSFORM_GLOB"] = "*.txt"
    env["TRANSFORM_RESUME"] = "False"
    
    process = subprocess.run(
        ["python", "transform.py", "--batch"],
        env=env,
        capture_output=True,
        text=True
    )
    print(process.stdout)
    print(process.stderr)

    # 處理英文文件 (使用純 Python 邏輯)
    print("處理英文文件（使用純 Python 函數）...")
    for file_path in INPUT_EN_DIR.glob("*.txt"):
        try:
            lines = read_transcript(str(file_path))
            # 直接呼叫 transform.py 裡的 reflow_to_paragraphs
            paragraphized_text = reflow_to_paragraphs("\n".join(lines))
            
            output_path = PARA_OUTPUT_DIR / (file_path.stem + "_paragraphized.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(paragraphized_text)
            print(f"處理完成：{output_path}")
        except Exception as e:
            print(f"英文文件處理失敗 {file_path.name}: {e}")

    print("--- 段落化程式執行完成 ---")
    time.sleep(2)

# --- 3. 執行翻譯程式 ---
def run_translator():
    print("--- 開始執行翻譯程式（監聽模式） ---")
    process = subprocess.Popen(
        [
            "python", "run_pipeline.py",
            "--para_dir", str(PARA_OUTPUT_DIR),
            "--out_zhtw", str(ZHTW_OUTPUT_DIR),
            "--out_zhcn", str(ZHCN_OUTPUT_DIR),
            "--out_en", str(EN_OUTPUT_DIR)
        ]
    )
    
    # 計算所有已分段的文件總數
    total_files = len(list(PARA_OUTPUT_DIR.glob("*.txt")))
    print(f"等待 {total_files} 個檔案翻譯完成...")
    while len(list(EN_OUTPUT_DIR.glob("*.en.txt"))) < total_files:
        print(f"已完成：{len(list(EN_OUTPUT_DIR.glob('*.en.txt')))}/{total_files}")
        time.sleep(5)
    
    process.terminate()
    print("--- 翻譯程式執行完成 ---")

# --- 4. 將翻譯結果寫回資料庫 ---
def import_to_db():
    print("--- 將翻譯結果寫回資料庫 ---")
    
    with MySQLConn(DB_NAME) as conn:
        with conn.cursor() as cursor:
            try:
                # 這裡需要新增欄位，如果還沒有的話
                cursor.execute("ALTER TABLE us_earnings_calls ADD COLUMN content_zhtw TEXT")
                cursor.execute("ALTER TABLE us_earnings_calls ADD COLUMN content_zhcn TEXT")
                cursor.execute("ALTER TABLE us_earnings_calls ADD COLUMN content_en TEXT")
                conn.commit()
            except pymysql.err.OperationalError:
                print("資料庫欄位已存在，跳過新增。")
                conn.rollback()

            en_files = EN_OUTPUT_DIR.glob("*.en.txt")
            for file_path in en_files:
                parts = file_path.stem.split('_')
                symbol, year, quarter = parts[0], parts[1], parts[2]
                
                content_en = file_path.read_text(encoding="utf-8")
                
                # 檢查中文翻譯文件是否存在
                zhtw_file = ZHTW_OUTPUT_DIR / f"{file_path.stem}.zhtw.txt"
                zhcn_file = ZHCN_OUTPUT_DIR / f"{file_path.stem}.zhcn.txt"
                
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
            
        conn.commit()
    print("--- 資料庫更新完成 ---")

# --- 執行主流程 ---
if __name__ == "__main__":
    export_from_db()
    run_paragraphizer()
    run_translator()
    import_to_db()
    print("--- 所有流程已完成 ---")