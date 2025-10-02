# -*- coding: utf-8 -*-
"""
tiqu_fixed.py
- 支持 Excel(.xlsx/.xls) 与 CSV 两种日志格式
- 自动识别得分列，自动推断 task_id 与可见行号的偏移
- 可选文本兜底匹配（question/题目 ↔ Question/问题）
- 最终将两对 (excel/csv, jsonl) 的零分样本合并去重输出

用法示例：
python tiqu_fixed.py \
  --excel1 评测/gpt4.1简答/evaluation_results_20250928_183724.csv --jsonl1 评测/0828所有vqa简答.jsonl \
  --excel2 评测/gpt4.1简答/evaluation_results_20250928_183916.csv --jsonl2 评测/0823所有简答.jsonl \
  --out gpt4.1简答.jsonl --fallback-text-match
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import pandas as pd

SCORE_CANDIDATES = ["score", "得分", "分数", "分", "Score", "SCORE"]
QUESTION_CANDIDATES_TAB = ["question", "Question", "题目", "问题", "题干"]
QUESTION_CANDIDATES_JSONL = ["Question", "question", "题目", "问题", "题干"]

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

def build_taskid_index(items: List[Dict]) -> Dict[int, Dict]:
    idx = {}
    for obj in items:
        if isinstance(obj, dict) and "task_id" in obj:
            try:
                idx[int(obj["task_id"])] = obj
            except Exception:
                pass
    return idx

def find_score_column(df: pd.DataFrame, hint: Optional[str] = None) -> str:
    if hint and hint in df.columns:
        return hint
    lowered = {c.lower(): c for c in df.columns}
    for cand in SCORE_CANDIDATES:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for c in df.columns:
        name = str(c)
        if any(tok in name for tok in ["score", "Score", "分"]):
            return c
    raise ValueError(f"未能识别得分列，请用 --score-col 指定。可选列：{list(df.columns)}")

def find_question_column(df: pd.DataFrame) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for cand in QUESTION_CANDIDATES_TAB:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None

def to_num(x):
    try:
        import pandas as pd
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

def choose_best_offset(zero_rows_onebased, task_ids_set, search_range=range(-5,6)) -> int:
    best_offset, best_hits = 0, -1
    for off in search_range:
        hits = sum((r + off) in task_ids_set for r in zero_rows_onebased)
        if hits > best_hits:
            best_hits, best_offset = hits, off
        elif hits == best_hits:
            pref = [0, -1, 1]
            if best_offset not in pref and off in pref:
                best_offset = off
            elif (best_offset not in pref and off not in pref and abs(off) < abs(best_offset)):
                best_offset = off
    return best_offset

def best_text_match(q: str, jsonl_items: List[Dict], json_q_keys: List[str], threshold=0.82) -> Optional[Dict]:
    q_norm = (q or "").strip()
    best_obj, best_ratio = None, 0.0
    for obj in jsonl_items:
        for k in json_q_keys:
            if k in obj:
                cand = str(obj[k]).strip()
                ratio = SequenceMatcher(None, q_norm, cand).ratio()
                if ratio > best_ratio:
                    best_ratio, best_obj = ratio, obj
    if best_ratio >= threshold:
        return best_obj
    return None

def read_table_auto(path: Path, sheet: str = "", encoding_hint: str = "") -> Tuple[pd.DataFrame, str]:
    """
    读取 Excel/CSV：
      - .xlsx/.xls -> Excel；优先 openpyxl（.xlsx），xlrd（.xls）；也可 engine="python" 兜底
      - .csv -> read_csv，优先 utf-8，失败则尝试 gbk
    返回 (DataFrame, 使用的工作表名或"")
    """
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        engine = None
        if suffix == ".xlsx":
            engine = "openpyxl"
        elif suffix == ".xls":
            engine = "xlrd"
        try:
            xl = pd.ExcelFile(path, engine=engine)
        except Exception:
            # 无法自动判断时，抛出更清晰的错误，提示安装引擎
            raise ValueError(
                f"无法读取Excel文件：{path}\n"
                f"请尝试安装依赖：pip install openpyxl xlrd\n"
                f"或者将日志另存为 CSV 后再试。"
            )
        # 选择表：优先用户提供的sheet，否则选第一个包含“得分列”的sheet
        if sheet and sheet in xl.sheet_names:
            df = xl.parse(sheet)
            return df, sheet
        chosen = xl.sheet_names[0]
        for s in xl.sheet_names:
            try_df = xl.parse(s)
            try:
                _ = find_score_column(try_df, None)
                chosen = s
                df = try_df
                return df, chosen
            except Exception:
                continue
        # 都没有得分列，仍返回第一张表
        df = xl.parse(chosen)
        return df, chosen
    else:
        # CSV
        encodings = [encoding_hint] if encoding_hint else []
        encodings += ["utf-8", "utf-8-sig", "gbk"]
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, engine="python")
                return df, ""
            except Exception as e:
                last_err = e
                continue
        raise ValueError(f"无法读取CSV文件：{path}\n最后错误：{last_err}\n可尝试指定 --encoding1/--encoding2 或另存为UTF-8编码。")

def extract_one_pair(excel_path: Path, jsonl_path: Path, sheet: str, score_col_hint: str,
                     enable_fallback: bool, text_threshold: float, encoding_hint: str = "") -> List[Dict]:
    jsonl_items = load_jsonl(jsonl_path)
    task_index = build_taskid_index(jsonl_items)
    task_id_set = set(task_index.keys())
    tag = jsonl_path.stem

    df, used_sheet = read_table_auto(excel_path, sheet=sheet, encoding_hint=encoding_hint)

    sc_col = find_score_column(df, score_col_hint or None)
    df["_score_num"] = df[sc_col].map(to_num)
    zero_df = df[df["_score_num"].fillna(0) == 0].copy()

    # 可见行号：表头为第1行，数据行索引+2
    zero_df["_visible_row_no"] = zero_df.index.to_series().astype(int) + 2
    zero_rows_onebased = zero_df["_visible_row_no"].tolist()
    best_offset = choose_best_offset(zero_rows_onebased, task_id_set, range(-5,6))

    q_col = find_question_column(df)
    json_q_keys = [k for k in QUESTION_CANDIDATES_JSONL]

    out_items = []
    for _, row in zero_df.iterrows():
        row_no = int(row["_visible_row_no"])
        candidate_tid = row_no + best_offset
        obj = task_index.get(candidate_tid)
        if obj is None and enable_fallback and q_col is not None:
            qx = row.get(q_col)
            if isinstance(qx, str) and qx.strip():
                obj = best_text_match(qx, jsonl_items, json_q_keys, threshold=text_threshold)
        if obj is not None:
            if isinstance(obj, dict):
                obj = dict(obj)
                obj.setdefault("__source", tag)
            out_items.append(obj)

    print(f"[INFO] 读取: {excel_path.name}  表/编码: {used_sheet or 'CSV'}  样本数: {len(df)}")
    print(f"[INFO] 零分行: {len(zero_df)}  偏移OFFSET: {best_offset}  导出: {len(out_items)}")

    return out_items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel1", required=True)
    ap.add_argument("--jsonl1", required=True)
    ap.add_argument("--excel2", required=True)
    ap.add_argument("--jsonl2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet1", default="")
    ap.add_argument("--sheet2", default="")
    ap.add_argument("--score-col1", default="")
    ap.add_argument("--score-col2", default="")
    ap.add_argument("--encoding1", default="")
    ap.add_argument("--encoding2", default="")
    ap.add_argument("--fallback-text-match", dest="fallback_text_match", action="store_true")
    ap.add_argument("--text-threshold", type=float, default=0.82)
    args = ap.parse_args()

    items1 = extract_one_pair(Path(args.excel1), Path(args.jsonl1), args.sheet1, args.score_col1,
                              args.fallback_text_match, args.text_threshold, encoding_hint=args.encoding1)
    items2 = extract_one_pair(Path(args.excel2), Path(args.jsonl2), args.sheet2, args.score_col2,
                              args.fallback_text_match, args.text_threshold, encoding_hint=args.encoding2)

    def key_for(obj):
        tag = obj.get("__source", "ds")
        if "task_id" in obj:
            try:
                return (tag, int(obj["task_id"]))
            except Exception:
                pass
        return (tag, json.dumps(obj, sort_keys=True, ensure_ascii=False))

    merged = {}
    for coll in [items1, items2]:
        for obj in coll:
            merged.setdefault(key_for(obj), obj)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in merged.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] 合并输出: {out_path}  共 {len(merged)} 条")

if __name__ == "__main__":
    main()
