import asyncio
import os
import aiohttp
from dotenv import load_dotenv

from prompts.base_templates import BASE_TEMPLATES
from prompts.improved_templates import IMPROVED_TEMPLATES
from optimizer.async_runner import run_all
from optimizer.evaluator import evaluate, extract_error_patterns

load_dotenv()

async def main():
    # Step 1. Merge all templates
    templates = BASE_TEMPLATES + IMPROVED_TEMPLATES

    print("🚀 Running BASE_TEMPLATES...")
    await run_all(templates=BASE_TEMPLATES)

    print("🚀 Running IMPROVED_TEMPLATES...")
    await run_all(templates=IMPROVED_TEMPLATES)

    # Step 2. Evaluate all predictions
    print("\n📊 Evaluating results...")
    memory = evaluate()

    print("\n📋 템플릿 성능 요약:")
    for m in memory:
        print(f"- {m['template_id']} | eval_count={m['eval_count']} | "
              f"recall={m['avg_recall']} | precision={m['avg_precision']} | f1={m['avg_f1']}")

    # Step 3. Save error cases
    extract_error_patterns()

    print("\n✅ summary.csv 저장 완료 → data/summary.csv")
    print("✅ 실패 문장 저장 완료 → data/errors.jsonl")
    print("✅ 실험 완료. improved_templates.py에 복붙하세요.")

if __name__ == "__main__":
    asyncio.run(main())