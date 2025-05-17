# optimizer/async_runner.py

import asyncio
import aiohttp
import random
import csv
import json
from tqdm import tqdm
from config import SAMPLE_SIZE
from engine.api_client import call_llm

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

def format_prompt(template, input_text):
    if isinstance(template, str):  # single-turn
        return [{"role": "user", "content": template.format(text=input_text)}]
    elif isinstance(template, list):  # multi-turn
        return [
            {
                "role": msg["role"],
                "content": msg["content"].format(text=input_text)
            } for msg in template
        ]
    else:
        raise ValueError(f"Unsupported template format: {type(template)}")

async def run_single(template_obj, row):
    messages = format_prompt(template_obj["template"], row["input"])
    await asyncio.sleep(0.3)  # 속도 조절
    result = await call_llm(messages)

    return {
        "template_id": template_obj["id"],
        "id": row["id"],
        "input": row["input"],
        "prediction": result,
        "target": row.get("target")
    }

async def run_all(train_path="data/test_with_answer.csv", out_path="data/results.jsonl", limit=100, templates=None):
    data = load_train_csv(train_path, limit)
    results = []

    tasks = [
        run_single(template, row)
        for row in data
        for template in (templates or [])
    ]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        if result:
            results.append(result)

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")