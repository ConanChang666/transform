# -*- coding: utf-8 -*-
"""
逐字稿 → 段落式逐字稿（保留每句、不摘要）
支援：
- .env 批次：自動掃資料夾、輸出到指定資料夾
- 單檔模式（保留原 CLI）
- 429/401 掛機等待重試（自動解析等待秒數 / 重新讀 .env）
- 多模型輪替（遇 429 換下一個）
"""

import os, re, sys, json, argparse, unicodedata, hashlib, time
from typing import List, Dict, Optional
from pathlib import Path

from dotenv import load_dotenv

try:
    from groq import Groq
    USE_SDK = True
except Exception:
    import requests
    USE_SDK = False

# ---------- 基礎 ----------
TIME_TAG_RE = re.compile(r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\]")
ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
ONLY_THANKS_RE = re.compile(r"^\s*(謝謝(大家)?|感謝(各位)?|thanks|thank you|感恩)\s*[。!\.!?]?\s*$", re.I)

def normalize_zh(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def light_cleanup(line: str, keep_time_tag: bool=False) -> str:
    s = line.rstrip("\n")
    if not keep_time_tag:
        s = TIME_TAG_RE.sub("", s)
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

# ---------- 雜訊清理（可選） ----------
def clean_noise_line(line: str) -> str:
    line = re.sub(r"(謝)\1{1,}", r"\1\1", line)
    line = re.sub(r"[\(（](掌聲|笑聲|喝采|鼓掌|音樂|旁白)[\)）]", "", line)
    return line.strip()

def read_transcript(path: str, keep_time_tag=False, light_clean=True,
                    clean_noise=True, max_thanks_keep: int = 1) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = normalize_zh(raw)
            if light_clean:
                raw = light_cleanup(raw, keep_time_tag=keep_time_tag)
            if clean_noise:
                raw = clean_noise_line(raw)
            if not raw.strip():
                continue
            if out and ONLY_THANKS_RE.match(raw) and ONLY_THANKS_RE.match(out[-1]):
                count = 1
                i = len(out) - 2
                while i >= 0 and ONLY_THANKS_RE.match(out[i]):
                    count += 1
                    i -= 1
                if count >= max_thanks_keep:
                    continue
            out.append(raw.strip())
    return out

def pack_chunks(lines: List[str], max_chars: int=5000) -> List[List[str]]:
    chunks, cur, acc = [], [], 0
    for ln in lines:
        L = len(ln) + 1
        if acc + L > max_chars and cur:
            chunks.append(cur)
            cur, acc = [ln], L
        else:
            cur.append(ln); acc += L
    if cur: chunks.append(cur)
    return chunks

# ---------- 覆蓋率 ----------
def _strip_all(s: str) -> str:
    return re.sub(r"\s+", "", s)

def coverage_check(src_lines: List[str], out_text: str) -> Dict[str, float]:
    def norm_punct(t: str) -> str:
        t = t.replace(",", "，").replace(".", "。").replace("?", "？").replace("!", "！").replace(";", "；").replace(":", "：")
        def fix_line(s: str) -> str:
            s = s.rstrip()
            if re.search(r"[\u4e00-\u9fff]", s) and not re.search(r"[。！？]$", s):
                s += "。"
            return s
        return "\n".join(fix_line(ln) for ln in t.split("\n"))
    src_text = "\n".join(src_lines)
    src_norm = norm_punct(src_text)
    out_norm = norm_punct(out_text)
    src_chars = len(_strip_all(src_norm))
    out_chars = len(_strip_all(out_norm))
    splitter = re.compile(r"[。！？\n]+")
    src_sent_n = len([x for x in splitter.split(src_norm) if x.strip()])
    out_sent_n = len([x for x in splitter.split(out_norm) if x.strip()])
    return {"char_ratio": out_chars / max(src_chars, 1),
            "sent_ratio": out_sent_n / max(src_sent_n, 1)}

# ---------- 標點後處理（可選） ----------
def ensure_zh_fullwidth_punct(text: str) -> str:
    def repl_punct(m):
        seg = m.group(0)
        seg = seg.replace(",", "，").replace(".", "。").replace("?", "？") \
                 .replace("!", "！").replace(";", "；").replace(":", "：")
        return seg
    text = re.sub(r"(?<=[\u4e00-\u9fff])[,\.?!;:]+(?=[\u4e00-\u9fff])", repl_punct, text)
    lines = text.split("\n")
    fixed = []
    for ln in lines:
        s = ln.rstrip()
        if s and ZH_CHAR_RE.search(s):
            if not re.search(r"[。！？）」】』]$", s):
                s += "。"
        fixed.append(s)
    return "\n".join(fixed)

# ---------- 語意分段重排 ----------
TOPIC_CUES = tuple([
    "首先","先","接著","再來","另外","另一方面","同時","此外","其次",
    "總結","最後","未來","展望","風險","策略","產能","良率","供應鏈",
    "市場","需求","產品","技術","氣冷","水冷","伺服器","財報","營收",
    "毛利","成本","資本支出","訂單","客戶","地緣政治","AI","HPC","Q&A",
    "問：","答：","主持人：","發言人：","分析師：","投資人："
])

def split_sentences_zh(text: str) -> List[str]:
    text = re.sub(r"\n+", " ", text)
    pat = re.compile(r"[^。！？!?]*[。！？!?]|[^。！？!?]+$")
    sents = [m.group(0).strip() for m in pat.finditer(text)]
    return [s for s in sents if s]

def is_topic_boundary(sent: str) -> bool:
    s = sent.strip()
    if any(s.startswith(cue) for cue in TOPIC_CUES):
        return True
    if re.match(r"^(主持人|發言人|分析師|投資人|問|答)[：:]", s):
        return True
    return False

def reflow_to_paragraphs(text: str, min_sent: int = 3, max_sent: int = 6) -> str:
    sents = split_sentences_zh(text)
    paras, buf = [], []
    for s in sents:
        if buf and is_topic_boundary(s) and len(buf) >= max(min_sent, 1):
            paras.append("".join(buf)); buf = [s]; continue
        buf.append(s)
        if len(buf) >= max_sent:
            paras.append("".join(buf)); buf = []
    if buf: paras.append("".join(buf))
    return "\n\n".join(paras)

# ---------- 工具：環境變數 ----------
def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default

# ---------- Prompt ----------
SYSTEM_PROMPT = """你是一位專業逐字稿整理編輯，任務是將「逐句切開的逐字稿」重排成「可閱讀的段落式逐字稿」。
**最重要規則：不得刪除、改寫、濃縮任何一句原文內容**。你可以：
- 調整標點與換行
- 合併多句同主題為自然段
- 在必要處加入最少量的連接標點（如頓號、逗號、句號），但不得新增實質內容
- 使用中文全形標點（。！？、；：）並於句末補齊適當標點
- 若偵測到說話者（如「主持人：」「發言人：」或姓名），保留其標示
**關鍵輸出格式**：
- 同一段落內不得逐句換行；句子之間直接連續書寫
- 僅能用「空行」分隔段落，每段建議 3–6 句，主題切換時才換段
輸出僅包含整理後的文字，不要加任何前後說明。"""

USER_PROMPT_TEMPLATE = """請將下列逐字稿內容改寫為「段落式逐字稿」，**不刪減任何一句話**、**不摘要**、**不改動專有名詞與數字**。
允許的操作只有：調整標點、換行與段落合併，維持原話語序與意思。請使用中文全形標點，並為中文句末補齊「。」「！」「？」等適當標點。
**同一段落內不得逐句換行，僅用「空行」分段；每段 3–6 句，遇主題切換再換段。**

【逐字稿片段（按原始順序）】
{chunk}
"""

# ---------- Groq 呼叫：429 直接換模型 ----------
RE_TRY_AGAIN_S  = re.compile(r"try again in ([0-9\.]+)s", re.I)
RE_TRY_AGAIN_MS = re.compile(r"try again in (\d+)m([0-9\.]+)s", re.I)

def _model_list_from_env() -> List[str]:
    txt = os.getenv("TRANSFORM_MODELS", "").strip()
    if txt:
        return [m.strip() for m in txt.split(",") if m.strip()]
    return [os.getenv("TRANSFORM_GROQ_MODEL", "llama-3.1-8b-instant")]

def call_groq(model: str, system_prompt: str, user_prompt: str,
              temperature: float=0.0, timeout: int=60, max_tokens: int=4096) -> str:
    WAIT_401 = _env_int("TRANSFORM_WAIT_ON_401", 600)
    WAIT_429_FALLBACK = _env_int("TRANSFORM_WAIT_ON_429", 90)
    WAIT_5XX = _env_int("TRANSFORM_WAIT_ON_5XX", 60)
    WAIT_MISC = _env_int("TRANSFORM_WAIT_ON_MISC", 30)

    url = "https://api.groq.com/openai/v1/chat/completions"

    cur_idx = 0
    models = _model_list_from_env()

    while True:
        load_dotenv(override=True)
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            print(f"[WARN] 缺少 GROQ_API_KEY，{WAIT_401}s 後重試…")
            time.sleep(WAIT_401)
            continue

        model = models[cur_idx % len(models)]

        try:
            if USE_SDK:
                client = Groq(api_key=api_key, timeout=timeout)
                resp = client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role":"system","content":system_prompt.strip()},
                              {"role":"user","content":user_prompt.strip()}],
                )
                return (resp.choices[0].message.content or "").strip()
            else:
                import requests
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role":"system","content":system_prompt.strip()},
                        {"role":"user","content":user_prompt.strip()},
                    ],
                }
                r = requests.post(url, headers=headers, json=payload, timeout=timeout)

                if r.status_code == 401:
                    print(f"[WARN] 401（Key 無效/不可用）。{WAIT_401}s 後重試…")
                    time.sleep(WAIT_401); continue

                if r.status_code == 429:
                    # 不睡，直接換下一個模型
                    msg = ""
                    try: msg = r.json().get("error", {}).get("message", "")
                    except: pass
                    print(f"[WARN] 429（{model}）：{msg} → 換下一個模型")
                    cur_idx += 1

                    # 全部模型都試完一輪仍 429 → 這時候才睡一下，再重來一輪
                    if cur_idx % len(models) == 0:
                        wait_s = None
                        ra = r.headers.get("Retry-After")
                        if ra:
                            try: wait_s = float(ra)
                            except: pass
                        if wait_s is None:
                            m = RE_TRY_AGAIN_MS.search(msg) or RE_TRY_AGAIN_S.search(msg)
                            if m and len(m.groups()) == 2:
                                wait_s = int(m.group(1))*60 + float(m.group(2))
                            elif m:
                                wait_s = float(m.group(1))
                        if wait_s is None:
                            wait_s = WAIT_429_FALLBACK
                        print(f"[INFO] 所有模型都滿了，睡 {wait_s:.1f}s 後再試一輪…")
                        time.sleep(wait_s)
                    continue

                if 500 <= r.status_code < 600:
                    print(f"[WARN] 服務端 {r.status_code}（{model}）。{WAIT_5XX}s 後重試…")
                    time.sleep(WAIT_5XX); continue

                r.raise_for_status()
                data = r.json()
                return (data["choices"][0]["message"]["content"] or "").strip()

        except Exception as e:
            print(f"[ERROR] {model} 呼叫失敗：{e}，{WAIT_MISC}s 後重試…")
            time.sleep(WAIT_MISC)

# ---------- checkpoint / 存檔 ----------
def load_ckpt(ckpt_path: str) -> Dict:
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": {}, "order": []}

def save_ckpt(ckpt_path: str, data: Dict):
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ckpt_path)

def append_final(output_path: str, chunk_text: str, first_append: bool):
    mode = "w" if first_append else "a"
    with open(output_path, mode, encoding="utf-8") as f:
        if not first_append:
            f.write("\n\n")
        f.write(chunk_text.strip())

def safe_write(path: str, text: str):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

# ---------- 主流程（單檔） ----------
def run(input_path: str, output_path: str, model: str,
        max_chars: int=5000, keep_time_tag: bool=False, light_clean: bool=True,
        resume: bool=True, per_chunk_dir: Optional[str]=None,
        clean_noise: bool=True, ensure_zh_punct_flag: bool=True,
        reflow: bool=True, min_sent_per_para: int=3, max_sent_per_para: int=6):

    lines = read_transcript(
        input_path, keep_time_tag=keep_time_tag, light_clean=light_clean,
        clean_noise=clean_noise, max_thanks_keep=1
    )
    if not lines:
        raise RuntimeError(f"讀不到內容：{input_path}")

    chunks = pack_chunks(lines, max_chars=max_chars)
    print(f"[INFO] {os.path.basename(input_path)}：總行數 {len(lines)}，分成 {len(chunks)} 片段。")

    ckpt_path = output_path + ".progress.json"
    ckpt = load_ckpt(ckpt_path)

    already_done = set(int(k) for k in ckpt.get("done", {}).keys())
    first_append = not os.path.exists(output_path) or (not resume)
    if not resume:
        if os.path.exists(output_path): os.remove(output_path)
        if os.path.exists(ckpt_path): os.remove(ckpt_path)
        ckpt = {"done": {}, "order": []}
        already_done = set()
        first_append = True

    for idx, seg in enumerate(chunks, 1):
        seg_text = "\n".join(seg)
        in_hash = hashlib.sha1(seg_text.encode("utf-8")).hexdigest()

        if idx in already_done and ckpt["done"].get(str(idx), {}).get("in_hash") == in_hash:
            print(f"[SKIP] 片段 {idx} 已完成，跳過。")
            continue

        user_prompt = USER_PROMPT_TEMPLATE.format(chunk=seg_text)
        print(f"[RUN ] 片段 {idx}/{len(chunks)} …")
        out = call_groq(
            model=model, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
            temperature=0.0, max_tokens=4096
        )

        if ensure_zh_punct_flag:
            out = ensure_zh_fullwidth_punct(out)
        if reflow:
            out = reflow_to_paragraphs(out, min_sent=min_sent_per_para, max_sent=max_sent_per_para)

        cov = coverage_check(seg, out)
        print(f"[INFO] 覆蓋率 char={cov['char_ratio']:.2%}, sent={cov['sent_ratio']:.2%}")

        if per_chunk_dir:
            safe_write(os.path.join(per_chunk_dir, f"chunk_{idx:04d}_OUTPUT.txt"), out)

        append_final(output_path, out, first_append)
        first_append = False

        ckpt["done"][str(idx)] = {"in_hash": in_hash,
                                  "out_hash": hashlib.sha1(out.encode("utf-8")).hexdigest(),
                                  "char_ratio": cov["char_ratio"], "sent_ratio": cov["sent_ratio"]}
        ckpt["order"].append(idx)
        save_ckpt(ckpt_path, ckpt)

        print(f"[OK  ] 片段 {idx} 完成。")

    print(f"[DONE] 輸出：{output_path}\n       進度檔：{ckpt_path}")

# ---------- 批次模式（讀 .env） ----------
def env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def run_batch_from_env():
    load_dotenv()

    in_dir = os.getenv("TRANSFORM_INPUT_DIR")
    out_dir = os.getenv("TRANSFORM_OUTPUT_DIR")
    if not in_dir or not out_dir:
        raise RuntimeError("請在 .env 設定 TRANSFORM_INPUT_DIR 與 TRANSFORM_OUTPUT_DIR")

    model = _model_list_from_env()[0]

    max_chars = int(os.getenv("TRANSFORM_MAX_CHARS", "5000"))
    resume = env_bool("TRANSFORM_RESUME", True)
    keep_time_tag = env_bool("TRANSFORM_KEEP_TIME_TAG", False)
    light_clean = env_bool("TRANSFORM_LIGHT_CLEAN", True)
    clean_noise = env_bool("TRANSFORM_CLEAN_NOISE", True)
    ensure_zh_punct_flag = env_bool("TRANSFORM_ENSURE_ZH_PUNCT", True)
    reflow = env_bool("TRANSFORM_REFLOW", True)
    min_sent = int(os.getenv("TRANSFORM_MIN_SENT", "3"))
    max_sent = int(os.getenv("TRANSFORM_MAX_SENT", "6"))
    per_chunk = env_bool("TRANSFORM_PER_CHUNK_DIR", True)
    glob_pat = os.getenv("TRANSFORM_GLOB", "*.txt")

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(Path(in_dir).glob(glob_pat))
    if not files:
        print(f"[WARN] 找不到檔案：{in_dir}/{glob_pat}")
        return

    print(f"[BATCH] 共 {len(files)} 個檔案。輸出到：{out_dir}")
    fail = 0
    for p in files:
        if not p.is_file(): continue
        stem = p.stem
        output_path = os.path.join(out_dir, f"{stem}_paragraphized.txt")
        per_chunk_dir = os.path.join(out_dir, "chunks", stem) if per_chunk else None
        if per_chunk_dir:
            os.makedirs(per_chunk_dir, exist_ok=True)

        try:
            run(
                input_path=str(p),
                output_path=output_path,
                model=model,
                max_chars=max_chars,
                keep_time_tag=keep_time_tag,
                light_clean=light_clean,
                resume=resume,
                per_chunk_dir=per_chunk_dir,
                clean_noise=clean_noise,
                ensure_zh_punct_flag=ensure_zh_punct_flag,
                reflow=reflow,
                min_sent_per_para=min_sent,
                max_sent_per_para=max_sent,
            )
        except Exception as e:
            fail += 1
            print(f"[ERROR] {p.name}: {e}")

    ok = len(files) - fail
    print(f"[BATCH DONE] 成功 {ok}，失敗 {fail}")

# ---------- 參數解析（保留單檔模式） ----------
def parse_args():
    p = argparse.ArgumentParser(description="逐字稿 → 段落式逐字稿（支援 .env 批次 + 多模型輪替）")
    p.add_argument("--input", help="單檔：輸入檔")
    p.add_argument("--output", help="單檔：輸出檔")
    p.add_argument("--model", default="llama-3.1-8b-instant")
    p.add_argument("--max_chars", type=int, default=5000)
    p.add_argument("--keep_time_tag", type=lambda x: str(x).lower()=="true", default=False)
    p.add_argument("--light_clean", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--resume", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--per_chunk_dir", default="", help="單檔：逐片輸出目錄")
    p.add_argument("--clean_noise", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--ensure_zh_punct", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--reflow", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--min_sent_per_para", type=int, default=3)
    p.add_argument("--max_sent_per_para", type=int, default=6)
    p.add_argument("--batch", action="store_true", help="啟用 .env 批次模式（不用再打參數）")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        if args.batch or (not args.input and not args.output):
            run_batch_from_env()
        else:
            if not args.input or not args.output:
                raise RuntimeError("單檔模式需要 --input 與 --output")
            load_dotenv()
            run(
                input_path=args.input,
                output_path=args.output,
                model=args.model,
                max_chars=args.max_chars,
                keep_time_tag=args.keep_time_tag,
                light_clean=args.light_clean,
                resume=args.resume,
                per_chunk_dir=(args.per_chunk_dir or None),
                clean_noise=args.clean_noise,
                ensure_zh_punct_flag=args.ensure_zh_punct,
                reflow=args.reflow,
                min_sent_per_para=args.min_sent_per_para,
                max_sent_per_para=args.max_sent_per_para,
            )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)