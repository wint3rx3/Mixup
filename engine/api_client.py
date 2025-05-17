# engine/api_client.py

import os
import asyncio
from itertools import cycle
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.upstage.ai/v1"
MODEL = "solar-pro"

# 🔑 최대 10개 API 키 로딩
API_KEYS = [os.getenv(f"UPSTAGE_API_KEY_{i}") for i in range(1, 11)]
API_KEYS = [key for key in API_KEYS if key]

# 🔁 client + semaphore 쌍 생성
clients = [
    (AsyncOpenAI(api_key=key, base_url=BASE_URL), asyncio.Semaphore(4))
    for key in API_KEYS
]

# ♻️ 순환자 생성 (round-robin 방식)
client_cycle = cycle(clients)

# ✅ 다중 키 기반 LLM 호출 함수 (기존과 동일한 인터페이스 유지)
async def call_llm(messages: list, retries: int = 3, delay: float = 1.2) -> str:
    client, semaphore = next(client_cycle)  # 다음 클라이언트/세마포어 가져오기

    async with semaphore:
        for attempt in range(retries):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e):
                    await asyncio.sleep(delay * (attempt + 1))  # 점진적 딜레이
                else:
                    break

    return "[ERROR] 호출 실패"
