# -*- coding: utf-8 -*-
# watch_and_translate.py
# 功能：監聽 PARA_DIR，新檔案出現（*_paragraphized.txt）→ 先繁體 (.zhtw.txt) 再英文 (.en.txt)
# 特點：增量處理、進度列（tqdm；若未安裝則自動略過）、嚴格 token 切塊（安全模式）、靜音 max_length 警告

import os, re, time, warnings
from pathlib import Path
from transformers import pipeline, AutoTokenizer

# --- 靜音 HuggingFace 的 max_length 類警告（僅訊息，不影響結果） ---
warnings.filterwarnings("ignore", message=".*max_length.*")

# --- 進度列（可選） ---
try:
    from tqdm import tqdm     # pip install tqdm
except Exception:
    tqdm = None

# ====== 參數 ======
PARA_DIR   = "/Users/fiiconan/Desktop/transform/file_complete_transform"
OUT_ZHTW   = "/Users/fiiconan/Desktop/transform/translate_to_tranditional_Chinese"
OUT_EN     = "/Users/fiiconan/Desktop/transform/translate_to_English"
PATTERN    = "*_paragraphized.txt"

SHOW_PROGRESS = True          # True 顯示 per-line 進度；未安裝 tqdm 會自動忽略
POLL_INTERVAL = 5             # 每幾秒掃描一次

# 安全模式：嚴格避免超過模型宣告長度 512（不再出錯，只可能有提示）
MAX_TOKENS = 500              # 每個送入模型的片段，encode 後的 token 上限
MAX_LENGTH = 512              # 翻譯時的 max_length
BATCH_SIZE = 16
MODEL_ZH_HANT_TO_EN = "HPLT/translate-zh_hant-en-v1.0-hplt_opus"
DEVICE = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1

# ====== OpenCC ======
def _get_opencc():
    try:
        from opencc import OpenCC
        _ = OpenCC("s2t.json")
        return OpenCC
    except Exception:
        from opencc import OpenCC
        return OpenCC
OpenCC = _get_opencc()
_conv_s2t = OpenCC("s2t"); _conv_t2s = OpenCC("t2s")

def _diff(a,b): return sum(1 for x,y in zip(a,b) if x!=y) + abs(len(a)-len(b))
def is_simplified_line(line: str) -> bool:
    if not line.strip(): return False
    to_s = _conv_t2s.convert(line); to_t = _conv_s2t.convert(line)
    return _diff(line,to_s) < _diff(line,to_t)

# ====== I/O ======
def read_lines(p: Path): return p.read_text(encoding="utf-8").splitlines()
def write_lines(p: Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")

# ====== 嚴格 token 切塊（保證 encode 後 ≤ MAX_TOKENS） ======
def smart_split_for_model(text: str, tokenizer, max_tokens: int = MAX_TOKENS):
    if not text.strip(): return []
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
            flush(); cur_ids.extend(sent_ids)
    flush()
    return chunks

# ====== 翻譯器 ======
def build_translator(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("translation", model=model_name, tokenizer=tok, device=DEVICE)
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

# ====== 繁→英（含進度列） ======
def translate_lines_to_en(lines, translator, tokenizer):
    out_lines = []
    iterator = enumerate(lines, 1)
    total = len(lines)
    if tqdm and SHOW_PROGRESS:
        iterator = tqdm(iterator, total=total, desc="To EN", unit="line")
    for _, ln in iterator:
        if not ln.strip():
            out_lines.append(ln); continue
        chunks = smart_split_for_model(ln, tokenizer, max_tokens=MAX_TOKENS)
        out_parts = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            res = translator(batch, max_length=MAX_LENGTH)
            outs = [r["translation_text"] if isinstance(r, dict) else r for r in res]
            out_parts.extend(outs)
        out_lines.append(" ".join(out_parts))
    return out_lines

# ====== 增量判斷 ======
def is_up_to_date(src: Path, dst: Path) -> bool:
    return dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime

# ====== 單檔處理 ======
def process_file(file: Path, out_zhtw_root: Path, out_en_root: Path, translator, tokenizer):
    out_zhtw = out_zhtw_root / (file.stem + ".zhtw.txt")
    out_en   = out_en_root   / (file.stem + ".en.txt")

    # 繁體
    if is_up_to_date(file, out_zhtw):
        print(f"[SKIP ZHTW] {file.name}")
        zhtw_lines = read_lines(out_zhtw)
    else:
        lines = read_lines(file)
        zhtw_lines = normalize_lines_to_traditional(lines)
        write_lines(out_zhtw, zhtw_lines)
        print(f"[ZHTW DONE] {file.name} -> {out_zhtw}")

    # 英文
    if is_up_to_date(out_zhtw, out_en):
        print(f"[SKIP EN ] {file.name}")
    else:
        en_lines = translate_lines_to_en(zhtw_lines, translator, tokenizer)
        write_lines(out_en, en_lines)
        print(f"[EN  DONE] {file.name} -> {out_en}")

# ====== 主監聽迴圈 ======
def main():
    para_root = Path(PARA_DIR)
    out_zhtw_root = Path(OUT_ZHTW); out_zhtw_root.mkdir(parents=True, exist_ok=True)
    out_en_root = Path(OUT_EN); out_en_root.mkdir(parents=True, exist_ok=True)

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
                process_file(f, out_zhtw_root, out_en_root, translator, tokenizer)
                processed[f] = mtime
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

if __name__ == "__main__":
    main()


'''
pip install transformers sentencepiece tqdm opencc-python-reimplemented
python3 watch_and_translate.py
'''