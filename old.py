import asyncio
import os
import json
import argparse
from dotenv import load_dotenv
from config import SAMPLE_SIZE

from prompts.base_templates import BASE_TEMPLATES
from prompts.improved_templates import IMPROVED_TEMPLATES
from optimizer.async_runner import run_all
from optimizer.evaluator import evaluate
from optimizer.error_extractor import extract_failed_cases
from optimizer.summary_writer import write_summary_csv

load_dotenv()

async def main_loop(sample_size: int = SAMPLE_SIZE, mode: str = "auto"):
    os.makedirs("data", exist_ok=True)
    results_all = []

    base_path = "data/results_base.jsonl"
    improve_path = "data/results_improve.jsonl"
    merged_path = "data/results.jsonl"

    if mode == "base":
        print("🚀 Running BASE_TEMPLATES only...")
        await run_all("data/train.csv", base_path, sample_size, BASE_TEMPLATES)

    elif mode == "improve":
        print("🚀 Running IMPROVED_TEMPLATES only...")
        await run_all("data/train.csv", improve_path, sample_size, IMPROVED_TEMPLATES)

    elif mode == "both":
        print("🚀 Running BOTH BASE and IMPROVED templates...")
        await run_all("data/train.csv", base_path, sample_size, BASE_TEMPLATES)
        await run_all("data/train.csv", improve_path, sample_size, IMPROVED_TEMPLATES)

    elif mode == "auto":
        run_base = not os.path.exists(base_path)
        run_improve = os.path.exists(base_path)
        if run_base:
            print("🚀 Running BASE_TEMPLATES...")
            await run_all("data/train.csv", base_path, sample_size, BASE_TEMPLATES)
        if run_improve:
            print("🚀 Running IMPROVED_TEMPLATES...")
            await run_all("data/train.csv", improve_path, sample_size, IMPROVED_TEMPLATES)
        if not run_base and not run_improve:
            print("⚠️ Skip: 이미 두 결과가 모두 있음")

    # 병합
    print("📎 Merging results...")
    for path in [base_path, improve_path]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                results_all.extend(json.loads(line) for line in f)

    with open(merged_path, "w", encoding="utf-8") as f:
        for item in results_all:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 평가 및 후처리
    print("\n📊 Evaluating results...")
    memory = evaluate()

    print("\n📋 템플릿 성능 요약:")
    for m in memory:
        print(
            f"- {m['template_id']} | eval_count={m['eval_count']} | "
            f"recall={m['avg_recall']} | precision={m['avg_precision']} | f1={m['f1']}"
        )

    extract_failed_cases()
    write_summary_csv()

    print("\n✅ 실험 완료. improved_templates.py에 복붙하세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="auto", choices=["base", "improve", "both", "auto"])
    args = parser.parse_args()

    asyncio.run(main_loop(mode=args.mode))
