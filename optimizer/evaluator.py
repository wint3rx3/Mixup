# optimizer/evaluator.py

import json
from collections import defaultdict
import os
import difflib
from config import RESULTS_PATH, MEMORY_PATH, RECALL_THRESHOLD

def lcs_length(a: str, b: str) -> int:
    # 진짜 LCS 계산 (DP)
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def compute_scores(pred: str, target: str):
    pred = pred.strip()
    target = target.strip()
    if not pred or not target:
        return 0, 0, 0

    lcs = lcs_length(pred, target)
    recall = lcs / len(target) if target else 0
    precision = lcs / len(pred) if pred else 0
    return lcs, recall, precision

def evaluate(results_path=RESULTS_PATH, memory_path=MEMORY_PATH):
    scores_by_template = defaultdict(list)

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = item["prediction"]
            target = item.get("target")
            if not target:
                continue
            lcs, recall, precision = compute_scores(pred, target)
            scores_by_template[item["template_id"]].append({
                "lcs": lcs,
                "recall": recall,
                "precision": precision,
                "id": item["id"]
            })

    memory = []
    for tid, results in scores_by_template.items():
        recall_avg = sum(r["recall"] for r in results) / len(results)
        precision_avg = sum(r["precision"] for r in results) / len(results)
        lcs_avg = sum(r["lcs"] for r in results) / len(results)

        memory.append({
            "template_id": tid,
            "eval_count": len(results),
            "avg_recall": round(recall_avg, 4),
            "avg_precision": round(precision_avg, 4),
            "avg_lcs": round(lcs_avg, 2)
        })

    with open(memory_path, "w", encoding="utf-8") as fout:
        for m in memory:
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")

    return memory

def extract_error_patterns(
    results_path=RESULTS_PATH,
    output_path="data/prompt_error_patterns.jsonl",
    recall_threshold=RECALL_THRESHOLD
):
    patterns = []

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            target = item.get("target")
            if not target or item["prediction"].startswith("[ERROR]"):
                continue
            lcs, recall, _ = compute_scores(item["prediction"], target)
            if recall < recall_threshold:
                error_type = classify_error(item["input"], item["prediction"])
                patterns.append({
                    "template_id": item["template_id"],
                    "source_id": item["id"],
                    "recall": round(recall, 3),
                    "error_type": error_type,
                    "example": f"{item['input']} → {target}"
                })

    with open(output_path, "w", encoding="utf-8") as fout:
        for p in patterns:
            fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            
def classify_error(src: str, pred: str) -> str:
    if "은 " in src or "는 " in src or "이 " in src:
        return "조사 오류"
    elif any(p in src for p in [".", "?", "!", ","]) and not any(p in pred for p in [".", "?", "!", ","]):
        return "문장 부호 누락"
    elif "  " in src or " " not in src:
        return "띄어쓰기 오류"
    else:
        return "기타"
