import os, re
from pathlib import Path
from datetime import datetime
import pymysql
from dotenv import load_dotenv

PAT = re.compile(r'^(?P<stock>[A-Za-z\.]{1,10}|\d{4,6})_(?P<date>\d{8})\.txt$', re.IGNORECASE)

def parse_filename(name: str):
    m = PAT.match(name)
    if not m:
        return None
    stock = m.group("stock")
    d = datetime.strptime(m.group("date"), "%Y%m%d").date()
    return stock, d

def main():
    load_dotenv()

    folder = os.getenv("INGEST_DIR", ".")
    apply_mode = os.getenv("INGEST_APPLY", "false").lower() == "true"

    files = Path(folder).glob("*.txt")
    rows = []
    for f in files:
        info = parse_filename(f.name)
        if not info:
            print(f"[SKIP] {f.name} 不符合命名規則")
            continue
        stock_id, d = info
        rows.append((f.name, stock_id, d.isoformat(), None))  # time_stamp=None

    print(f"解析成功 {len(rows)} 筆")

    if not apply_mode:
        for r in rows[:10]:
            print("[DRY-RUN]", r)
        print("※ 現在是 dry-run，若要真正寫入，在 .env 設 INGEST_APPLY=true")
        return

    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST","127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT","3306")),
        user=os.getenv("MYSQL_USER","root"),
        password=os.getenv("MYSQL_PASSWORD",""),
        database=os.getenv("MYSQL_DB","mydb"),
        charset="utf8mb4",
        autocommit=False
    )

    sql = f"""
        INSERT INTO {os.getenv("MYSQL_TABLE","earnings_call_transcripts")}
        (original_filename, stock_id, `date`, time_stamp)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE original_filename = VALUES(original_filename)
    """

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    conn.close()
    print("寫入完成")

if __name__ == "__main__":
    main()