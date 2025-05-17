# optimizer/evaluator.py

import json
from collections import defaultdict
import os
from config import RESULTS_PATH, MEMORY_PATH, RECALL_THRESHOLD

def tokenize(text):
    if not text:
        return []
    return str(text).split()

def lcs_table(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i][j + 1], L[i + 1][j])
    return L

def find_lcs(X, Y):
    L = lcs_table(X, Y)
    i, j = len(X), len(Y)
    lcs = []
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs[::-1]

def find_differences_with_offsets(original, corrected):
    original_tokens = tokenize(original)
    corrected_tokens = tokenize(corrected)
    lcs = find_lcs(original_tokens, corrected_tokens)

    orig_index = corr_index = lcs_index = 0
    differences = []

    while orig_index < len(original_tokens) or corr_index < len(corrected_tokens):
        orig_diff, corr_diff = [], []
        orig_start, corr_start = orig_index, corr_index

        while orig_index < len(original_tokens) and (lcs_index >= len(lcs) or original_tokens[orig_index] != lcs[lcs_index]):
            orig_diff.append(original_tokens[orig_index])
            orig_index += 1
        while corr_index < len(corrected_tokens) and (lcs_index >= len(lcs) or corrected_tokens[corr_index] != lcs[lcs_index]):
            corr_diff.append(corrected_tokens[corr_index])
            corr_index += 1

        if orig_diff or corr_diff:
            differences.append((' '.join(orig_diff), ' '.join(corr_diff), orig_start, orig_index, corr_start, corr_index))

        if lcs_index < len(lcs):
            lcs_index += 1
            orig_index += 1
            corr_index += 1

    return differences

def evaluate(results_path=RESULTS_PATH, memory_path=MEMORY_PATH):
    scores_by_template = defaultdict(list)

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = item["prediction"]
            target = item.get("target")
            original = item["input"]
            if not target:
                continue

            diffs_og = find_differences_with_offsets(original, target)
            diffs_op = find_differences_with_offsets(original, pred)

            og_idx = op_idx = 0
            tp = fp = fm = fr = 0

            while True:
                if og_idx >= len(diffs_og) and op_idx >= len(diffs_op):
                    break
                if og_idx >= len(diffs_og):
                    fr += 1
                    op_idx += 1
                    continue
                if op_idx >= len(diffs_op):
                    fm += 1
                    og_idx += 1
                    continue
                if diffs_og[og_idx][2] == diffs_op[op_idx][2]:
                    if diffs_og[og_idx][1] == diffs_op[op_idx][1]:
                        tp += 1
                    else:
                        fp += 1
                    og_idx += 1
                    op_idx += 1
                elif diffs_og[og_idx][2] < diffs_op[op_idx][2]:
                    fm += 1
                    og_idx += 1
                else:
                    fr += 1
                    op_idx += 1

            recall = tp / (tp + fp + fm) if (tp + fp + fm) > 0 else 0.0
            precision = tp / (tp + fp + fr) if (tp + fp + fr) > 0 else 0.0

            scores_by_template[item["template_id"]].append({
                "recall": recall,
                "precision": precision,
                "id": item["id"]
            })

    memory = []
    for tid, results in scores_by_template.items():
        recall_avg = sum(r["recall"] for r in results) / len(results)
        precision_avg = sum(r["precision"] for r in results) / len(results)
        f1 = 2 * recall_avg * precision_avg / (recall_avg + precision_avg) if (recall_avg + precision_avg) > 0 else 0.0

        memory.append({
            "template_id": tid,
            "eval_count": len(results),
            "avg_recall": round(recall_avg, 4),
            "avg_precision": round(precision_avg, 4),
            "f1": round(f1, 4)  # ✅ 추가!
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
            pred = item["prediction"]
            target = item.get("target")
            if not target or pred.startswith("[ERROR]"):
                continue

            diffs_og = find_differences_with_offsets(item["input"], target)
            diffs_op = find_differences_with_offsets(item["input"], pred)

            og_idx = op_idx = 0
            tp = 0

            while og_idx < len(diffs_og) and op_idx < len(diffs_op):
                if diffs_og[og_idx][2] == diffs_op[op_idx][2] and diffs_og[og_idx][1] == diffs_op[op_idx][1]:
                    tp += 1
                og_idx += 1
                op_idx += 1

            recall = tp / len(diffs_og) if diffs_og else 0
            if recall < recall_threshold:
                error_type = classify_error(item["input"], pred)
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
