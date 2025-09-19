# -*- coding: utf-8 -*-
"""
run_pipeline.py（監聽 + 翻譯整合版）
功能：監聽 PARA_DIR，新檔（*_paragraphized.txt）→ 產出：
  - 繁體 .zhtw.txt
  - 簡體 .zhcn.txt（由繁體轉換）
  - 英文 .en.txt（繁體→英翻譯）

依賴：
  pip install transformers sentencepiece tqdm opencc-python-reimplemented torch

用法（主流程會這樣叫）：
  python run_pipeline.py \
    --para_dir <PARA_DIR> \
    --out_zhtw <OUT_ZHTW> \
    --out_zhcn <OUT_ZHCN> \
    --out_en   <OUT_EN>

也可直接跑（走預設值）：
  python run_pipeline.py
"""

import os
import re
import time
import warnings
import argparse
from pathlib import Path

# --- 靜音 HuggingFace 的 max_length 類警告（僅訊息，不影響結果） ---
warnings.filterwarnings("ignore", message=".*max_length.*")

# --- 進度列（可選） ---
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # 無 tqdm 時自動關閉進度列

from transformers import pipeline, AutoTokenizer  # type: ignore

# ====== 預設參數（可被命令列覆蓋） ======
DEFAULT_PARA_DIR = "/Users/fiiconan/Desktop/transform/file_complete_transform"
DEFAULT_OUT_ZHTW = "/Users/fiiconan/Desktop/transform/translate_to_tranditional_Chinese"
DEFAULT_OUT_ZHCN = "/Users/fiiconan/Desktop/transform/translate_to_simplified_Chinese"
DEFAULT_OUT_EN   = "/Users/fiiconan/Desktop/transform/translate_to_English"
PATTERN          = "*_paragraphized.txt"

SHOW_PROGRESS = True          # True 顯示 per-line 進度；未安裝 tqdm 會自動忽略
POLL_INTERVAL = 5             # 每幾秒掃描一次

# 安全模式：嚴格避免超過模型宣告長度 512（不再出錯，只可能有提示）
MAX_TOKENS = 500              # 每個送入模型的片段，encode 後的 token 上限
MAX_LENGTH = 512              # 翻譯時的 max_length
BATCH_SIZE = 16
MODEL_ZH_HANT_TO_EN = "HPLT/translate-zh_hant-en-v1.0-hplt_opus"

# ====== OpenCC ======
def _get_opencc():
    try:
        from opencc import OpenCC  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenCC 載入失敗。請先安裝：pip install opencc-python-reimplemented"
        ) from e
    # 這裡務必用 's2t' / 't2s'（不要加 .json）
    try:
        _ = OpenCC("s2t")
    except Exception as e:
        raise RuntimeError(
            "OpenCC 初始化失敗，檢查安裝與資源檔（config 目錄）是否完整。"
        ) from e
    return OpenCC

OpenCC = _get_opencc()
_conv_s2t = OpenCC("s2t")
_conv_t2s = OpenCC("t2s")

# ====== 工具 ======
def _diff(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))

def is_simplified_line(line: str) -> bool:
    if not line.strip():
        return False
    to_s = _conv_t2s.convert(line)
    to_t = _conv_s2t.convert(line)
    return _diff(line, to_s) < _diff(line, to_t)

def read_lines(p: Path):
    return p.read_text(encoding="utf-8").splitlines()

def write_lines(p: Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")

# ====== 嚴格 token 切塊（保證 encode 後 ≤ MAX_TOKENS） ======
def smart_split_for_model(text: str, tokenizer, max_tokens: int = MAX_TOKENS):
    if not text.strip():
        return []
    # 先按句號類分隔；之後仍過長則按 token 硬切
    sentences = re.split(r'(?<=[。．\.!?！？])\s*', text)
    sentences = [s for s in sentences if s.strip()]

    chunks, cur_ids = [], []

    def flush():
        if cur_ids:
            chunks.append(tokenizer.decode(cur_ids, skip_special_tokens=True))
            cur_ids.clear()

    for sent in sentences:
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        if len(sent_ids) > max_tokens:
            # 單句超長：按 token 硬切
            start = 0
            while start < len(sent_ids):
                end = min(start + max_tokens, len(sent_ids))
                flush()
                chunks.append(tokenizer.decode(sent_ids[start:end], skip_special_tokens=True))
                start = end
            continue
        if len(cur_ids) + len(sent_ids) <= max_tokens:
            cur_ids.extend(sent_ids)
        else:
            flush()
            cur_ids.extend(sent_ids)

    flush()
    return chunks

# ====== 翻譯器 ======
def pick_device_index() -> int:
    """
    盡量安全地挑選 device：
      - 有 torch.cuda 可用 → 0
      - 否則 → -1（CPU）
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1

def build_translator(model_name: str):
    device_idx = pick_device_index()
    tok = AutoTokenizer.from_pretrained(model_name)
    # 保持相容用法（transformers 新版可改用 device_map="auto"）
    pipe = pipeline("translation", model=model_name, tokenizer=tok, device=device_idx)
    return pipe, tok

# ====== 簡→繁（含進度列） ======
def normalize_lines_to_traditional(lines):
    out = []
    iterator = enumerate(lines, 1)
    total = len(lines)
    if tqdm and SHOW_PROGRESS:
        iterator = tqdm(iterator, total=total, desc="To ZHTW", unit="line")
    for _, ln in iterator:
        out.append(_conv_s2t.convert(ln) if is_simplified_line(ln) else ln)
    return out

# ====== 繁→簡（含進度列） ======
def normalize_lines_to_simplified(lines):
    out = []
    iterator = enumerate(lines, 1)
    total = len(lines)
    if tqdm and SHOW_PROGRESS:
        iterator = tqdm(iterator, total=total, desc="To ZHCN", unit="line")
    for _, ln in iterator:
        out.append(_conv_t2s.convert(ln) if ln.strip() else ln)
    return out

# ====== 繁→英（含進度列） ======
def translate_lines_to_en(lines, translator, tokenizer):
    out_lines = []
    iterator = enumerate(lines, 1)
    total = len(lines)
    if tqdm and SHOW_PROGRESS:
        iterator = tqdm(iterator, total=total, desc="To EN", unit="line")
    for _, ln in iterator:
        if not ln.strip():
            out_lines.append(ln)
            continue
        chunks = smart_split_for_model(ln, tokenizer, max_tokens=MAX_TOKENS)
        out_parts = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            res = translator(batch, max_length=MAX_LENGTH)
            outs = [r["translation_text"] if isinstance(r, dict) else r for r in res]
            out_parts.extend(outs)
        out_lines.append(" ".join(out_parts))
    return out_lines

# ====== 增量判斷 ======
def is_up_to_date(src: Path, dst: Path) -> bool:
    return dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime

# ====== 單檔處理 ======
def process_file(file: Path, out_zhtw_root: Path, out_zhcn_root: Path, out_en_root: Path, translator, tokenizer):
    out_zhtw = out_zhtw_root / (file.stem + ".zhtw.txt")
    out_zhcn = out_zhcn_root / (file.stem + ".zhcn.txt")
    out_en   = out_en_root   / (file.stem + ".en.txt")

    # 1) 繁體（以原始段落檔為基準做增量）
    if is_up_to_date(file, out_zhtw):
        print(f"[SKIP ZHTW] {file.name}")
        zhtw_lines = read_lines(out_zhtw)
    else:
        lines = read_lines(file)
        zhtw_lines = normalize_lines_to_traditional(lines)
        write_lines(out_zhtw, zhtw_lines)
        print(f"[ZHTW DONE] {file.name} -> {out_zhtw}")

    # 2) 簡體（以繁中檔為基準做增量，確保一致）
    if is_up_to_date(out_zhtw, out_zhcn):
        print(f"[SKIP ZHCN] {file.name}")
    else:
        zhcn_lines = normalize_lines_to_simplified(zhtw_lines)
        write_lines(out_zhcn, zhcn_lines)
        print(f"[ZHCN DONE] {file.name} -> {out_zhcn}")

    # 3) 英文（仍以繁中為來源，因模型為 zh_hant→en）
    if is_up_to_date(out_zhtw, out_en):
        print(f"[SKIP EN  ] {file.name}")
    else:
        en_lines = translate_lines_to_en(zhtw_lines, translator, tokenizer)
        write_lines(out_en, en_lines)
        print(f"[EN  DONE] {file.name} -> {out_en}")

# ====== 主監聽迴圈 ======
def watch_loop(para_root: Path, out_zhtw_root: Path, out_zhcn_root: Path, out_en_root: Path):
    translator, tokenizer = build_translator(MODEL_ZH_HANT_TO_EN)

    print(f"[INFO] Watching {para_root} for {PATTERN}")
    processed = {}

    try:
        while True:
            files = sorted(para_root.glob(PATTERN))
            for f in files:
                try:
                    mtime = f.stat().st_mtime
                except FileNotFoundError:
                    continue
                # 同一 mtime 跳過
                if processed.get(f) == mtime:
                    continue
                process_file(f, out_zhtw_root, out_zhcn_root, out_en_root, translator, tokenizer)
                processed[f] = mtime
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

# ====== CLI ======
def parse_args():
    ap = argparse.ArgumentParser(description="Watch para_dir and translate to zhtw/zhcn/en")
    ap.add_argument("--para_dir", default=DEFAULT_PARA_DIR, help="Paragraphized input directory")
    ap.add_argument("--out_zhtw", default=DEFAULT_OUT_ZHTW, help="Output directory for Traditional Chinese")
    ap.add_argument("--out_zhcn", default=DEFAULT_OUT_ZHCN, help="Output directory for Simplified Chinese")
    ap.add_argument("--out_en",   default=DEFAULT_OUT_EN,   help="Output directory for English")
    return ap.parse_args()

def main():
    args = parse_args()

    para_root = Path(args.para_dir)
    out_zhtw_root = Path(args.out_zhtw); out_zhtw_root.mkdir(parents=True, exist_ok=True)
    out_zhcn_root = Path(args.out_zhcn); out_zhcn_root.mkdir(parents=True, exist_ok=True)
    out_en_root   = Path(args.out_en);   out_en_root.mkdir(parents=True, exist_ok=True)

    watch_loop(para_root, out_zhtw_root, out_zhcn_root, out_en_root)

if __name__ == "__main__":
    main()
