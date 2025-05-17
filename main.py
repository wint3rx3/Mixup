# main.py

import asyncio
import os
from dotenv import load_dotenv
from config import SAMPLE_SIZE

from prompts.base_templates import BASE_TEMPLATES
from prompts.improved_templates import IMPROVED_TEMPLATES
from optimizer.async_runner import run_all
from optimizer.evaluator import evaluate
from optimizer.error_extractor import extract_failed_cases
from optimizer.summary_writer import write_summary_csv

load_dotenv()

async def main_loop(sample_size: int = SAMPLE_SIZE):
    os.makedirs("data", exist_ok=True)

    results_all = []

    # ✅ BASE_TEMPLATES 실행 여부 판단
    if not os.path.exists("data/results_base.jsonl"):
        print("🚀 Running BASE_TEMPLATES...")
        await run_all(
            train_path="data/train.csv",
            out_path="data/results_base.jsonl",
            limit=sample_size,
            templates=BASE_TEMPLATES
        )
    else:
        print("⏩ Skipping BASE_TEMPLATES (already exists)")

    # ✅ ALWAYS run improved templates
    print("🚀 Running IMPROVED_TEMPLATES...")
    await run_all(
        train_path="data/train.csv",
        out_path="data/results_improve.jsonl",
        limit=sample_size,
        templates=IMPROVED_TEMPLATES
    )

    # ✅ 결과 통합
    print("📎 Merging results...")
    for path in ["data/results_base.jsonl", "data/results_improve.jsonl"]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                results_all.extend(json.loads(line) for line in f)

    with open("data/results.jsonl", "w", encoding="utf-8") as f:
        for item in results_all:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ✅ 평가 + 후처리
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

    print("\n✅ 실험 완료. ChatGPT로 결과 분석 후 improved_templates.py에 복붙하세요.")

if __name__ == "__main__":
    import json
    asyncio.run(main_loop())
