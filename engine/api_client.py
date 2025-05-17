# engine/api_client.py

import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
BASE_URL = "https://api.upstage.ai/v1"
MODEL = "solar-pro"

# OpenAI SDK 방식 (Upstage 공식 가이드 준수)
client = AsyncOpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url=BASE_URL
)

async def call_llm(messages: list, retries: int = 3, delay: float = 1.2) -> str:
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                await asyncio.sleep(delay)
            else:
                break
    return "[ERROR] 호출 실패"