# optimizer/async_runner.py

import asyncio
import aiohttp
import random
import csv
import json
from tqdm import tqdm
from config import SAMPLE_SIZE
from prompts.template_format import format_prompt
from engine.api_client import call_llm

semaphore = asyncio.Semaphore(5)

def load_train_csv(filepath: str, limit: int = 100, shuffle: bool = True):
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = list(csv.DictReader(f))
        if shuffle:
            random.shuffle(reader)
        return [
            {
                "id": row["id"],
                "input": row["err_sentence"],
                "target": row.get("cor_sentence")
            }
            for row in reader[:limit]
        ]

async def run_single(template_obj, row):
    async with semaphore:
        messages = format_prompt(template_obj["template"], row["input"])
        await asyncio.sleep(1.3)  # ✅ 호출 간 딜레이 삽입 (최대 46 req/min 수준)
        try:
            result = await call_llm(messages)
        except Exception as e:
            print(f"❌ LLM 호출 실패: {e}")
            return None

        return {
            "template_id": template_obj["id"],
            "id": row["id"],
            "input": row["input"],
            "prediction": result.strip() if result else "[ERROR]",
            "target": row.get("target")
        }

async def run_all(train_path="data/test_with_answer.csv", out_path="data/results.jsonl", limit=100, templates=None):
    data = load_train_csv(train_path, limit)
    results = []

    async with aiohttp.ClientSession():
        tasks = [
            run_single(template, row)
            for row in data
            for template in templates or []
        ]

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await f
            if result:
                results.append(result)

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")