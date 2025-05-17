# optimizer/async_runner.py

import asyncio
import aiohttp
import random
import csv
import json
from tqdm import tqdm
from config import SAMPLE_SIZE

from prompts.base_templates import BASE_TEMPLATES
from prompts.template_format import format_prompt
from engine.api_client import call_llm  # Upstage Solar API에 맞춘 wrapper

def load_train_csv(filepath: str, limit: int = 100, shuffle: bool = True):
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = list(csv.DictReader(f))
        if shuffle:
            random.shuffle(reader)
        rows = [
            {
                "id": row["id"],
                "input": row["err_sentence"],
                "target": row.get("cor_sentence")
            }
            for row in reader[:limit]
        ]
    return rows

async def run_single(template_obj, row):
    messages = format_prompt(template_obj["template"], row["input"])
    try:
        result = await call_llm(messages)
        result = result.strip()
    except Exception:
        return None  # 오류 결과 제외

    return {
        "template_id": template_obj["id"],
        "id": row["id"],
        "input": row["input"],
        "prediction": result,
        "target": row.get("target", None)
    }

async def run_all(train_path="data/train.csv", out_path="data/results.jsonl", limit=100, templates=None):
    data = load_train_csv(train_path, limit)

    async with aiohttp.ClientSession():  # session 제거됨
        tasks = [
            run_single(template, row)
            for row in data
            for template in templates or []
        ]

        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await f
            if result:  # 오류가 아닌 것만 저장
                results.append(result)

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")