# -*- coding: utf-8 -*-
"""
extract_zero_by_question.py

按需提取逻辑：
1) 从 CSV/TSV 表格中找出“得分为 0”的行；
2) 取这些行的 question 字段文本；
3) 到原 JSONL 文件中，用 Question 字段做**文本匹配（默认严格相等）**；
4) 匹配成功的整行 JSON 对象写入一个新的 JSONL；
5) 末尾打印匹配成功数目（以及未匹配数目）。

用法示例：
python extract_zero_by_question.py \
  --csv eval.csv \
  --jsonl data.jsonl \
  --out zero_match.jsonl

可选参数：
  --score-col        明确指定分数列名（不指定将自动识别常见列名）
  --question-col     CSV 中问题列名（默认自动识别常见列名）
  --json-question-key  JSONL 中问题键名（默认自动识别常见键名）
  --encoding         CSV 文件编码（默认自动尝试 utf-8/utf-8-sig/gbk）
  --sep              CSV 分隔符（默认自动推断，或设置逗号/制表符等）
  --fuzzy            启用相似度匹配（SequenceMatcher），默认关闭（严格相等）
  --threshold 0.90   配合 --fuzzy 使用，设定最小相似度阈值（默认 0.90）
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import pandas as pd

SCORE_CANDIDATES = ["score", "得分", "分数", "分", "Score", "SCORE"]
QUESTION_CANDIDATES_TAB = ["question", "Question", "题目", "问题", "题干"]
QUESTION_CANDIDATES_JSONL = ["Question", "question", "题目", "问题", "题干"]

def read_table(path: Path, encoding_hint: str = "", sep_hint: str = "") -> pd.DataFrame:
    # 自动尝试编码
    encodings = [encoding_hint] if encoding_hint else []
    encodings += ["utf-8", "utf-8-sig", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            # 分隔符：优先用户指定，否则让 pandas 猜测（engine="python" + sep=None 会自动推断）
            if sep_hint:
                df = pd.read_csv(path, encoding=enc, engine="python", sep=sep_hint)
            else:
                df = pd.read_csv(path, encoding=enc, engine="python")
            return df
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"无法读取CSV/TSV：{path}\n最后错误：{last_err}\n可尝试指定 --encoding 或 --sep。")

def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data

def find_score_column(df: pd.DataFrame, hint: Optional[str] = None) -> str:
    if hint and hint in df.columns:
        return hint
    lowered = {c.lower(): c for c in df.columns}
    for cand in SCORE_CANDIDATES:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    # 模糊包含“score/分”字样
    for c in df.columns:
        name = str(c)
        if any(tok in name for tok in ["score", "Score", "分"]):
            return c
    raise ValueError(f"未能识别得分列，请用 --score-col 指定。可选列：{list(df.columns)}")

def find_question_column(df: pd.DataFrame, hint: Optional[str] = None) -> str:
    if hint and hint in df.columns:
        return hint
    lowered = {c.lower(): c for c in df.columns}
    for cand in QUESTION_CANDIDATES_TAB:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise ValueError(f"未能识别问题列，请用 --question-col 指定。可选列：{list(df.columns)}")

def find_json_question_key(items: List[Dict], hint: Optional[str] = None) -> str:
    if hint:
        return hint
    # 在样本中自动探测常见键
    for cand in QUESTION_CANDIDATES_JSONL:
        for obj in items[:50]:
            if isinstance(obj, dict) and cand in obj:
                return cand
    # 兜底
    return "Question"

def to_num(x):
    try:
        if pd.isna(x):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x)
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if m:
            return float(m.group())
    except Exception:
        return None
    return None

def normalize_text(s: str) -> str:
    # 严格匹配前，做最小规范化：去首尾空白、统一空白、全角空格->半角
    if s is None:
        return ""
    s = str(s).strip()
    # 全角空格替换
    s = s.replace("\u3000", " ")
    # 统一多空白为单空格
    s = re.sub(r"\s+", " ", s)
    return s

def build_exact_index(items: List[Dict], key: str) -> Dict[str, Dict]:
    idx = {}
    for obj in items:
        if not isinstance(obj, dict):
            continue
        val = obj.get(key, None)
        if val is None:
            continue
        idx[normalize_text(val)] = obj
    return idx

def best_fuzzy(q: str, items: List[Dict], key: str, threshold: float) -> Optional[Dict]:
    qn = normalize_text(q)
    best, best_r = None, 0.0
    for obj in items:
        if key not in obj:
            continue
        cand = normalize_text(obj[key])
        r = SequenceMatcher(None, qn, cand).ratio()
        if r > best_r:
            best_r, best = r, obj
    if best_r >= threshold:
        return best
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="评测结果 CSV/TSV 文件路径")
    ap.add_argument("--jsonl", required=True, help="原始 JSONL 路径")
    ap.add_argument("--out", required=True, help="输出 JSONL 路径")
    ap.add_argument("--score-col", default="", help="CSV 的得分列名（默认自动识别）")
    ap.add_argument("--question-col", default="", help="CSV 的问题列名（默认自动识别）")
    ap.add_argument("--json-question-key", default="", help="JSONL 的问题键名（默认自动识别）")
    ap.add_argument("--encoding", default="", help="CSV 编码（默认自动尝试）")
    ap.add_argument("--sep", default="", help="CSV 分隔符，如 ',' 或 '\\t'（默认自动推断）")
    ap.add_argument("--fuzzy", action="store_true", help="开启模糊匹配（默认关闭，严格相等）")
    ap.add_argument("--threshold", type=float, default=0.90, help="模糊匹配的最小相似度阈值")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out)

    # 读取 CSV/TSV
    df = read_table(csv_path, encoding_hint=args.encoding, sep_hint=args.sep)

    # 列识别
    score_col = find_score_column(df, args.score_col or None)
    q_col = find_question_column(df, args.question_col or None)

    # 过滤“得分为 0”的行
    df["_score_num"] = df[score_col].map(to_num)
    zero_df = df[df["_score_num"].fillna(0) == 0].copy()

    # 读取 JSONL & 识别问题键名
    items = load_jsonl(jsonl_path)
    json_q_key = find_json_question_key(items, args.json_question_key or None)

    # 构建精确匹配索引
    exact_index = build_exact_index(items, json_q_key)

    # 逐行匹配
    matched = []
    unmatched_examples = []
    for _, row in zero_df.iterrows():
        q_text = normalize_text(row[q_col])
        obj = exact_index.get(q_text)
        if obj is None and args.fuzzy:
            obj = best_fuzzy(q_text, items, json_q_key, args.threshold)
        if obj is not None:
            matched.append(obj)
        else:
            if len(unmatched_examples) < 5:
                unmatched_examples.append(q_text)

    # 写出
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in matched:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 打印统计
    total_zero = len(zero_df)
    print(f"[INFO] 零分行数：{total_zero}")
    print(f"[INFO] 匹配成功：{len(matched)} 条；未匹配：{total_zero - len(matched)} 条")
    if unmatched_examples:
        print("[HINT] 前若干未匹配示例：")
        for x in unmatched_examples:
            print("  -", x)

if __name__ == "__main__":
    main()
