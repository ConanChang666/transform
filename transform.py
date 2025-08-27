# -*- coding: utf-8 -*-
"""
逐字稿 → 段落式逐字稿（保留每句、不摘要）
Groq API 版｜逐片儲存、續跑、覆蓋率顯示、中文標點＋「謝謝」雜訊清理＋語意分段重排

用法：
python3 transform.py \
  --input 1102_20250318.txt \
  --output 1102_20250318_paragraphized.txt \
  --model llama-3.1-8b-instant \
  --max_chars 5000 \
  --resume true \
  --per_chunk_dir 1102_chunks \
  --clean_noise true \
  --ensure_zh_punct true \
  --reflow true \
  --min_sent_per_para 3 \
  --max_sent_per_para 6
"""

import os, re, sys, json, argparse, unicodedata, hashlib, time
from typing import List, Dict
from dotenv import load_dotenv

try:
    from groq import Groq
    USE_SDK = True
except Exception:
    import requests
    USE_SDK = False

# ---------- 基礎 ----------
TIME_TAG_RE = re.compile(r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\]")
ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")  # 是否含中文
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
    # 將「謝謝謝謝謝謝」→「謝謝」
    line = re.sub(r"(謝)\1{1,}", r"\1\1", line)
    # 去掉無內容括號標註
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
            # 控制連續 only-「謝謝」型句子的最大保留次數
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
    # 標點規整（讓兩邊用同一口徑）
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
    """
    保守地把中文語境中的英文標點換成全形；補齊中文句末句號。
    不動英文片段。
    """
    def repl_punct(m):
        seg = m.group(0)
        seg = seg.replace(",", "，").replace(".", "。").replace("?", "？") \
                 .replace("!", "！").replace(";", "；").replace(":", "：")
        return seg

    # 修正：前後都是中文字時，置換中間的半形標點
    text = re.sub(r"(?<=[\u4e00-\u9fff])[,\.?!;:]+(?=[\u4e00-\u9fff])", repl_punct, text)

    # 句末若是中文且沒有終止標點，補上「。」
    lines = text.split("\n")
    fixed = []
    for ln in lines:
        s = ln.rstrip()
        if s and ZH_CHAR_RE.search(s):
            if not re.search(r"[。！？）」】』]$", s):
                s += "。"
        fixed.append(s)
    return "\n".join(fixed)

# ---------- 語意分段重排（新） ----------
TOPIC_CUES = tuple([
    "首先", "先", "接著", "再來", "另外", "另一方面", "同時", "此外", "其次",
    "總結", "最後", "未來", "展望", "風險", "策略", "產能", "良率", "供應鏈",
    "市場", "需求", "產品", "技術", "氣冷", "水冷", "伺服器", "財報", "營收",
    "毛利", "成本", "資本支出", "訂單", "客戶", "地緣政治", "AI", "HPC", "Q&A",
    "問：", "答：", "主持人：", "發言人：", "分析師：", "投資人："
])

def split_sentences_zh(text: str) -> List[str]:
    """
    以中文句末標點（。！？）與英文?!，保留標點，拆成句子列表。
    不改動內容。
    """
    # 將換行視為空白，避免把每句拆成獨立行
    text = re.sub(r"\n+", " ", text)
    pat = re.compile(r"[^。！？!?]*[。！？!?]|[^。！？!?]+$")
    sents = [m.group(0).strip() for m in pat.finditer(text)]
    return [s for s in sents if s]

def is_topic_boundary(sent: str) -> bool:
    s = sent.strip()
    if any(s.startswith(cue) for cue in TOPIC_CUES):
        return True
    # 若出現明確說話者標籤，也視為邊界
    if re.match(r"^(主持人|發言人|分析師|投資人|問|答)[：:]", s):
        return True
    return False

def reflow_to_paragraphs(text: str, min_sent: int = 3, max_sent: int = 6) -> str:
    """
    只重排換行與段落：把句子串成 3–6 句的自然段；遇到主題邊界或說話者標籤即換段。
    不改變任何句子內容。
    """
    sents = split_sentences_zh(text)
    paras, buf = [], []

    for i, s in enumerate(sents):
        # 如果是明顯的主題邊界且目前段落達到最少句數，先換段
        if buf and is_topic_boundary(s) and len(buf) >= max(min_sent, 1):
            paras.append("".join(buf))  # 中文段落直接連在一起更自然
            buf = [s]
            continue

        buf.append(s)

        # 依句數上限換段
        if len(buf) >= max_sent:
            paras.append("".join(buf))
            buf = []

    if buf:
        paras.append("".join(buf))

    # 段落以「空行」分隔；段內不另起行
    return "\n\n".join(paras)

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

# ---------- Groq ----------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def call_groq(model: str, system_prompt: str, user_prompt: str,
              temperature: float=0.0, timeout: int=60, max_tokens: int=4096) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("缺少 GROQ_API_KEY（請設在環境變數或 .env）")
    if USE_SDK:
        client = Groq(api_key=api_key, timeout=timeout)
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role":"system","content":system_prompt.strip()},
                      {"role":"user","content":user_prompt.strip()}],
        )
        return (resp.choices[0].message.content or "").strip()
    else:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
        payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                   "messages": [{"role":"system","content":system_prompt.strip()},
                                {"role":"user","content":user_prompt.strip()}]}
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

# ---------- checkpoint / 儲存 ----------
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

# ---------- 主流程 ----------
def run(input_path: str, output_path: str, model: str,
        max_chars: int=5000, keep_time_tag: bool=False, light_clean: bool=True,
        resume: bool=True, per_chunk_dir: str=None,
        clean_noise: bool=True, ensure_zh_punct_flag: bool=True,
        reflow: bool=True, min_sent_per_para: int=3, max_sent_per_para: int=6):

    load_dotenv()
    lines = read_transcript(
        input_path, keep_time_tag=keep_time_tag, light_clean=light_clean,
        clean_noise=clean_noise, max_thanks_keep=1
    )
    if not lines:
        raise RuntimeError("讀不到內容，請檢查檔案或編碼。")

    chunks = pack_chunks(lines, max_chars=max_chars)
    print(f"[INFO] 總行數 {len(lines)}，分成 {len(chunks)} 個片段。")

    ckpt_path = output_path + ".progress.json"
    ckpt = load_ckpt(ckpt_path)

    already_done = set(int(k) for k in ckpt.get("done", {}).keys())
    print(f"[INFO] 已完成片段：{sorted(list(already_done))}" if already_done else "[INFO] 尚無已完成片段")

    first_append = not os.path.exists(output_path) or (not resume)
    if not resume:
        if os.path.exists(output_path): os.remove(output_path)
        if os.path.exists(ckpt_path): os.remove(ckpt_path)
        ckpt = {"done": {}, "order": []}
        already_done = set()
        first_append = True

    for idx, seg in enumerate(chunks, 1):
        seg_text = "\n".join(seg)
        in_hash = sha1(seg_text)

        if idx in already_done and ckpt["done"].get(str(idx), {}).get("in_hash") == in_hash:
            print(f"[SKIP] 片段 {idx} 已完成，跳過。")
            continue

        user_prompt = USER_PROMPT_TEMPLATE.format(chunk=seg_text)
        print(f"[RUN ] 片段 {idx}/{len(chunks)} …")

        out = call_groq(model=model, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, temperature=0.0, max_tokens=4096)

        # 標點保險
        if ensure_zh_punct_flag:
            out = ensure_zh_fullwidth_punct(out)

        # 語意分段重排
        if reflow:
            out = reflow_to_paragraphs(out, min_sent=min_sent_per_para, max_sent=max_sent_per_para)

        # 覆蓋率顯示
        cov = coverage_check(seg, out)
        print(f"[INFO] 覆蓋率：char={cov['char_ratio']:.2%}, sent={cov['sent_ratio']:.2%}")

        # 逐片存檔
        if per_chunk_dir:
            safe_write(os.path.join(per_chunk_dir, f"chunk_{idx:04d}_OUTPUT.txt"), out)

        append_final(output_path, out, first_append)
        first_append = False

        ckpt["done"][str(idx)] = {"in_hash": in_hash, "out_hash": sha1(out),
                                  "char_ratio": cov["char_ratio"], "sent_ratio": cov["sent_ratio"]}
        ckpt["order"].append(idx)
        save_ckpt(ckpt_path, ckpt)

        print(f"[OK  ] 片段 {idx} 完成。")

    print(f"[DONE] 全部完成。輸出：{output_path}\n       進度檔：{ckpt_path}")

def parse_args():
    p = argparse.ArgumentParser(description="逐字稿 → 段落式逐字稿（逐片儲存＋續跑＋覆蓋率＋中文標點＋雜訊清理＋語意分段）")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="llama-3.1-8b-instant")
    p.add_argument("--max_chars", type=int, default=5000)
    p.add_argument("--keep_time_tag", type=lambda x: str(x).lower()=="true", default=False)
    p.add_argument("--light_clean", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--resume", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--per_chunk_dir", default="", help="若提供路徑，將逐片輸出各自存檔")
    p.add_argument("--clean_noise", type=lambda x: str(x).lower()=="true", default=True,
                   help="啟用輕雜訊清理（壓縮連續『謝謝』、移除(掌聲)等標註）")
    p.add_argument("--ensure_zh_punct", type=lambda x: str(x).lower()=="true", default=True,
                   help="模型輸出後保守地補齊中文標點")
    p.add_argument("--reflow", type=lambda x: str(x).lower()=="true", default=True,
                   help="根據意思將句子重排為 3–6 句的段落，僅以空行分隔")
    p.add_argument("--min_sent_per_para", type=int, default=3)
    p.add_argument("--max_sent_per_para", type=int, default=6)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
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
