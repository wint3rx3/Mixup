# optimizer/view_results.py

import os
import json
import pandas as pd
from optimizer.reinforce_graph import ReinforceGraph

def flatten_metrics(node: dict) -> dict:
    result = {
        "id": node.get("id"),
        "template": node.get("template"),
        "generation": node.get("generation"),
        "parent_ids": ", ".join(node.get("parent_ids", []))
    }

    for stage in ["train_recall", "valid_recall"]:
        metrics = node.get(stage, {})
        for key, value in metrics.items():
            result[f"{stage}_{key}"] = value
    return result

def export_to_csv(path: str = "optimizer/results.csv"):
    graph = ReinforceGraph()
    all_nodes = graph.load_all()
    flat = [flatten_metrics(n) for n in all_nodes]
    df = pd.DataFrame(flat)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장 완료 → {path}")

def display_all_generations(max_gen: int = 5):
    graph = ReinforceGraph()
    for gen in range(max_gen):
        nodes = graph.get_generation(gen)
        print(f"\n🔷 Generation {gen} | {len(nodes)} prompts")
        for node in nodes:
            valid = node.get('valid_recall', {})
            recall = valid.get('recall', 0)
            precision = valid.get('precision', 0)
            print(f"- {node['id']} | R: {recall:.2f} | P: {precision:.2f}")
            print(f"  → {node['template']}\n")

def display_topk(k: int = 5, by: str = "valid_recall.recall"):
    graph = ReinforceGraph()
    top_nodes = graph.get_topk(k=k, by=by)
    print(f"\n🏆 Top-{k} Prompts by '{by}'\n")
    for i, node in enumerate(top_nodes):
        valid = node.get("valid_recall", {})
        recall = valid.get("recall", 0)
        precision = valid.get("precision", 0)
        print(f"{i+1}. {node['id']} | R: {recall:.2f} | P: {precision:.2f}")
        print(f"   {node['template']}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5, help="상위 몇 개를 출력할지")
    parser.add_argument("--by", type=str, default="valid_recall.recall", help="정렬 기준 필드")
    parser.add_argument("--gens", type=int, default=3, help="세대 출력 최대 개수")
    parser.add_argument("--csv", action="store_true", help="CSV 저장 여부")

    args = parser.parse_args()

    display_all_generations(args.gens)
    display_topk(k=args.topk, by=args.by)

    if args.csv:
        export_to_csv()