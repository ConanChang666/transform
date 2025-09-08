#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from dotenv import load_dotenv
import pymysql

NAME_RE = re.compile(
    r""" ^
         (?P<stock>[A-Za-z0-9]+)_
         (?P<date>\d{8})
         (?:_[^.]+)* 
         \.(?P<lang>zhtw|en)\.txt
       $ """,
    re.X | re.I,
)

LANG_MAP = {"zhtw": "zh-TW", "en": "en"}

def parse_key(filename: str) -> Tuple[str, str, str]:
    m = NAME_RE.match(filename)
    if not m:
        raise ValueError(f"無法解析檔名：{filename}")
    stock = m.group("stock")
    date8 = m.group("date")
    lang = m.group("lang").lower()
    date_ymd = f"{date8[0:4]}-{date8[4:6]}-{date8[6:8]}"
    return stock, date_ymd, lang

def read_text(p: Path) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt

def split_paragraphs(txt: Optional[str]) -> List[str]:
    if not txt:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]

def get_db_conn():
    load_dotenv()
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DB", "mydb"),
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )

# ---------------- SQL ----------------
GET_PARENT_SQL = """
SELECT id FROM earnings_call_transcripts
WHERE stock_id = %s AND call_date = %s
LIMIT 1;
"""

INSERT_PARENT_SQL = """
INSERT INTO earnings_call_transcripts (stock_id, call_date, created_at, updated_at)
VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
"""

UPSERT_CHILD_SQL = """
INSERT INTO earnings_call_transcript_translations
  (transcript_id, lang, transcript)
VALUES
  (%s, %s, CAST(%s AS JSON))
ON DUPLICATE KEY UPDATE
  transcript = VALUES(transcript);
"""

def get_or_create_transcript_id(conn, *, stock_id: str, call_date: str) -> int:
    with conn.cursor() as cur:
        cur.execute(GET_PARENT_SQL, (stock_id, call_date))
        row = cur.fetchone()
        if row:
            return int(row["id"])
        cur.execute(INSERT_PARENT_SQL, (stock_id, call_date))
        return int(cur.lastrowid)

def upsert_translation(conn, *, transcript_id: int, lang_code: str, paragraphs: List[str]) -> None:
    json_array = json.dumps(paragraphs, ensure_ascii=False)
    with conn.cursor() as cur:
        cur.execute(UPSERT_CHILD_SQL, (transcript_id, lang_code, json_array))

def build_pairs(zh_dir: Path, en_dir: Path) -> Dict[Tuple[str, str], Dict[str, Path]]:
    pairs: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for p in zh_dir.glob("*.txt"):
        try:
            stock, date_ymd, lang = parse_key(p.name)
            if lang == "zhtw":
                pairs.setdefault((stock, date_ymd), {})["zhtw"] = p
        except Exception:
            continue
    for p in en_dir.glob("*.txt"):
        try:
            stock, date_ymd, lang = parse_key(p.name)
            if lang == "en":
                pairs.setdefault((stock, date_ymd), {})["en"] = p
        except Exception:
            continue
    return pairs

def main():
    ap = argparse.ArgumentParser(description="Insert transcripts as JSON arrays into MySQL.")
    ap.add_argument("--zh-dir", required=True, help="中文檔資料夾")
    ap.add_argument("--en-dir", required=True, help="英文檔資料夾")
    args = ap.parse_args()

    zh_dir = Path(args.zh_dir).expanduser().resolve()
    en_dir = Path(args.en_dir).expanduser().resolve()

    pairs = build_pairs(zh_dir, en_dir)
    if not pairs:
        print("找不到任何可解析的檔案")
        return

    conn = get_db_conn()
    try:
        for (stock, date_ymd), files in sorted(pairs.items()):
            zh_path = files.get("zhtw")
            en_path = files.get("en")

            zh_segments = split_paragraphs(read_text(zh_path)) if zh_path else []
            en_segments = split_paragraphs(read_text(en_path)) if en_path else []

            try:
                transcript_id = get_or_create_transcript_id(conn, stock_id=stock, call_date=date_ymd)

                if zh_segments:
                    upsert_translation(conn, transcript_id=transcript_id, lang_code=LANG_MAP["zhtw"], paragraphs=zh_segments)

                if en_segments:
                    upsert_translation(conn, transcript_id=transcript_id, lang_code=LANG_MAP["en"], paragraphs=en_segments)

                conn.commit()
                print(f"[OK] {stock} {date_ymd} zh:{len(zh_segments)}段 en:{len(en_segments)}段 (id={transcript_id})")
            except Exception as e:
                conn.rollback()
                print(f"[FAIL] {stock} {date_ymd}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()

'''
python insert_to_db.py \
  --zh-dir "/Users/fiiconan/Desktop/transform/translate_to_tranditional_Chinese" \
  --en-dir "/Users/fiiconan/Desktop/transform/translate_to_English" \
'''