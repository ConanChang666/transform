# -*- coding: utf-8 -*-
"""
run_pipeline.py（監聽 + 翻譯整合版）
同時支援：
- 中文來源（--mode zh）：中文 → (先繁) → 簡 → （可選）英
- 英文來源（--mode en）：英文 → 中 → (先繁) → 簡 → （可選）英

用法（例）：
  python run_pipeline.py \
    --para_dir temp/paragraphized/zh \
    --out_zhtw final/zhtw \
    --out_zhcn final/zhcn \
    --out_en   final/en \
    --mode zh \
    --pattern "*.txt" \
    --force_s2t           # 可選
    # --skip_en           # 可選：若暫時不需要英文成品

  # 英文來源（只產繁/簡）
  python run_pipeline.py \
    --para_dir temp/paragraphized/en \
    --out_zhtw final/zhtw \
    --out_zhcn final/zhcn \
    --out_en   final/en \
    --mode en \
    --pattern "*.txt" \
    --skip_en
"""

import os
import re
import time
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore", message=".*max_length.*")

# tqdm（可選）
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

from transformers import pipeline, AutoTokenizer  # type: ignore

# ====== 預設參數（可被 CLI 覆蓋） ======
DEFAULT_PARA_DIR = "./temp/paragraphized/zh"
DEFAULT_OUT_ZHTW = "./final/zhtw"
DEFAULT_OUT_ZHCN = "./final/zhcn"
DEFAULT_OUT_EN   = "./final/en"

DEFAULT_PATTERN  = "*_paragraphized.txt"
SHOW_PROGRESS    = True
POLL_INTERVAL    = 5

# 安全切塊
MAX_TOKENS = 500
MAX_LENGTH = 512
BATCH_SIZE = 16

# 模型
MODEL_ZH_HANT_TO_EN = "HPLT/translate-zh_hant-en-v1.0-hplt_opus"
MODEL_EN_TO_ZH      = "Helsinki-NLP/opus-mt-en-zh"   # 輕量穩定；輸出多為簡體

# ====== OpenCC ======
def _get_opencc():
    try:
        from opencc import OpenCC  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCC 載入失敗。請先安裝：pip install opencc-python-reimplemented") from e
    try:
        _ = OpenCC("s2t")  # 確認資源
    except Exception as e:
        raise RuntimeError("OpenCC 初始化失敗，請檢查安裝與資源檔（config 目錄）是否完整。") from e
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

# ====== 嚴格 token 切塊 ======
def smart_split_for_model(text: str, tokenizer, max_tokens: int = MAX_TOKENS):
    if not text.strip():
        return []
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

# ====== 翻譯器建置與裝置選擇 ======
def pick_device_index() -> int:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1  # CPU

def build_translator(model_name: str):
    device_idx = pick_device_index()
    tok = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("translation", model=model_name, tokenizer=tok, device=device_idx)
    return pipe, tok

# ====== 行級正規化（簡→繁 / 繁→簡） ======
def normalize_lines_to_traditional(lines, force_s2t=False):
    out = []
    it = enumerate(lines, 1)
    total = len(lines)
    if tqdm and SHOW_PROGRESS:
        it = tqdm(it, total=total, desc="To ZHTW", unit="line")
    for _, ln in it:
        if not ln.strip():
            out.append(ln); continue
        if force_s2t:
            out.append(_conv_s2t.convert(ln))
        else:
            out.append(_conv_s2t.convert(ln) if is_simplified_line(ln) else ln)
    return out

def normalize_lines_to_simplified(lines):
    out = []
    it = enumerate(lines, 1)
    total = len(lines)
    if tqdm and SHOW_PROGRESS:
        it = tqdm(it, total=total, desc="To ZHCN", unit="line")
    for _, ln in it:
        out.append(_conv_t2s.convert(ln) if ln.strip() else ln)
    return out

# ====== 繁→英 ======
def translate_lines_to_en(lines, translator, tokenizer):
    out_lines = []
    it = enumerate(lines, 1)
    total = len(lines)
    if tqdm and SHOW_PROGRESS:
        it = tqdm(it, total=total, desc="To EN", unit="line")
    for _, ln in it:
        if not ln.strip():
            out_lines.append(ln)
            continue
        chunks = smart_split_for_model(ln, tokenizer, max_tokens=MAX_TOKENS)
        parts = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            res = translator(batch, max_length=MAX_LENGTH)
            parts.extend([r["translation_text"] if isinstance(r, dict) else r for r in res])
        out_lines.append(" ".join(parts))
    return out_lines

# ====== 英→中 ======
def translate_lines_en_to_zh(lines, translator, tokenizer):
    out_lines = []
    it = enumerate(lines, 1)
    total = len(lines)
    if tqdm and SHOW_PROGRESS:
        it = tqdm(it, total=total, desc="EN→ZH", unit="line")
    for _, ln in it:
        if not ln.strip():
            out_lines.append(ln)
            continue
        chunks = smart_split_for_model(ln, tokenizer, max_tokens=MAX_TOKENS)
        parts = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            res = translator(batch, max_length=MAX_LENGTH)
            parts.extend([r["translation_text"] if isinstance(r, dict) else r for r in res])
        out_lines.append(" ".join(parts))
    return out_lines

# ====== 時戳判斷 ======
def is_up_to_date(src: Path, dst: Path) -> bool:
    return dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime

# ====== 單檔處理 ======
def process_file(
    file: Path,
    out_zhtw_root: Path,
    out_zhcn_root: Path,
    out_en_root: Path,
    translator,
    tokenizer,
    mode: str = "zh",
    skip_en: bool = False,
    force_s2t: bool = False,
):
    out_zhtw = out_zhtw_root / (file.stem + ".zhtw.txt")
    out_zhcn = out_zhcn_root / (file.stem + ".zhcn.txt")
    out_en   = out_en_root   / (file.stem + ".en.txt")

    if mode == "zh":
        # 1) 以中文檔為來源：先繁
        if is_up_to_date(file, out_zhtw):
            print(f"[SKIP ZHTW] {file.name}")
            zhtw_lines = read_lines(out_zhtw)
        else:
            lines = read_lines(file)
            zhtw_lines = normalize_lines_to_traditional(lines, force_s2t=force_s2t)
            write_lines(out_zhtw, zhtw_lines)
            print(f"[ZHTW DONE] {file.name} -> {out_zhtw}")

        # 2) 由繁 → 簡
        if is_up_to_date(out_zhtw, out_zhcn):
            print(f"[SKIP ZHCN] {file.name}")
        else:
            zhcn_lines = normalize_lines_to_simplified(zhtw_lines)
            write_lines(out_zhcn, zhcn_lines)
            print(f"[ZHCN DONE] {file.name} -> {out_zhcn}")

        # 3) 由繁 → 英（可選）
        if not skip_en:
            if is_up_to_date(out_zhtw, out_en):
                print(f"[SKIP EN  ] {file.name}")
            else:
                # zh_hant → en
                en_lines = translate_lines_to_en(zhtw_lines, translator, tokenizer)
                write_lines(out_en, en_lines)
                print(f"[EN  DONE] {file.name} -> {out_en}")

    else:
        # mode == "en"
        # 1) EN → ZH（多為簡）
        lines_en_src = read_lines(file)
        zh_lines = translate_lines_en_to_zh(lines_en_src, translator, tokenizer)

        # 2) 把 ZH 統一轉為繁作為「真源」
        zhtw_lines = [_conv_s2t.convert(x) if x.strip() else x for x in zh_lines]

        # 3) 寫繁中
        if is_up_to_date(file, out_zhtw):
            print(f"[SKIP ZHTW] {file.name}")
        else:
            write_lines(out_zhtw, zhtw_lines)
            print(f"[ZHTW DONE] {file.name} -> {out_zhtw}")

        # 4) 由繁 → 簡
        if is_up_to_date(out_zhtw, out_zhcn):
            print(f"[SKIP ZHCN] {file.name}")
        else:
            zhcn_lines = [_conv_t2s.convert(x) if x.strip() else x for x in zhtw_lines]
            write_lines(out_zhcn, zhcn_lines)
            print(f"[ZHCN DONE] {file.name} -> {out_zhcn}")

        # 5) 英文輸出（預設跳過）
        if not skip_en:
            if is_up_to_date(file, out_en):
                print(f"[SKIP EN  ] {file.name}")
            else:
                write_lines(out_en, lines_en_src)  # 直接用原英文（也可改成 EN→EN 清理）
                print(f"[EN  DONE] {file.name} -> {out_en}")

# ====== 監聽主迴圈 ======
def watch_loop(
    para_root: Path,
    out_zhtw_root: Path,
    out_zhcn_root: Path,
    out_en_root: Path,
    pattern: str,
    mode: str,
    skip_en: bool,
    force_s2t: bool,
):
    # 依模式挑模型
    if mode == "zh":
        translator, tokenizer = build_translator(MODEL_ZH_HANT_TO_EN)
    else:
        translator, tokenizer = build_translator(MODEL_EN_TO_ZH)

    print(f"[INFO] Watching {para_root} for pattern='{pattern}' (mode={mode})")
    processed = {}

    try:
        while True:
            files = sorted(para_root.glob(pattern))
            for f in files:
                try:
                    mtime = f.stat().st_mtime
                except FileNotFoundError:
                    continue
                if processed.get(f) == mtime:
                    continue
                process_file(
                    f, out_zhtw_root, out_zhcn_root, out_en_root,
                    translator, tokenizer,
                    mode=mode, skip_en=skip_en, force_s2t=force_s2t
                )
                processed[f] = mtime
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

# ====== CLI ======
def parse_args():
    ap = argparse.ArgumentParser(description="Watch para_dir and translate to zhtw/zhcn[/en]")
    ap.add_argument("--para_dir", default=DEFAULT_PARA_DIR, help="Paragraphized input directory")
    ap.add_argument("--out_zhtw", default=DEFAULT_OUT_ZHTW, help="Output dir for Traditional Chinese")
    ap.add_argument("--out_zhcn", default=DEFAULT_OUT_ZHCN, help="Output dir for Simplified Chinese")
    ap.add_argument("--out_en",   default=DEFAULT_OUT_EN,   help="Output dir for English")
    ap.add_argument("--pattern",  default=DEFAULT_PATTERN,  help="Glob pattern for input files")
    ap.add_argument("--mode",     choices=["zh","en"], default="zh", help="Source mode: zh=Chinese, en=English")
    ap.add_argument("--skip_en",  action="store_true", help="Skip English outputs")
    ap.add_argument("--force_s2t", action="store_true", help="Force full S->T on Chinese source")
    return ap.parse_args()

def main():
    args = parse_args()
    para_root     = Path(args.para_dir)
    out_zhtw_root = Path(args.out_zhtw); out_zhtw_root.mkdir(parents=True, exist_ok=True)
    out_zhcn_root = Path(args.out_zhcn); out_zhcn_root.mkdir(parents=True, exist_ok=True)
    out_en_root   = Path(args.out_en);   out_en_root.mkdir(parents=True, exist_ok=True)

    watch_loop(
        para_root, out_zhtw_root, out_zhcn_root, out_en_root,
        pattern=args.pattern, mode=args.mode, skip_en=args.skip_en, force_s2t=args.force_s2t
    )

if __name__ == "__main__":
    main()
