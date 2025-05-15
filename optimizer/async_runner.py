# optimizer/async_runner.py

import asyncio
import httpx
import pandas as pd
from typing import Dict, List
from submission.utils.metrics import evaluate_correction
from submission.config import ExperimentConfig

class AsyncExperimentRunner:
    def __init__(self, config: ExperimentConfig, api_key: str, template: str):
        self.config = config
        self.api_key = api_key
        self.template = template
        self.api_url = config.api_url
        self.model = config.model
        # 더 이상 여기서 Semaphore 생성하지 않습니다.

    def _make_prompt(self, text: str) -> str:
        return self.template.format(text=text)

    async def _call_api(
        self,
        prompt: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        # 루프 내에서 생성된 semaphore 사용
        async with semaphore:
            for attempt in range(3):  # 최대 3회 재시도
                try:
                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        json=data,
                        timeout=10.0
                    )
                    if response.status_code == 429:
                        wait = 2 ** attempt  # 1s, 2s, 4s
                        print(f"[429 Too Many Requests] {attempt+1}/3회. {wait}초 대기 후 재시도...")
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"[Attempt {attempt+1}/3] API 호출 실패: {e}")
                    await asyncio.sleep(2 * (attempt + 1))
            return "[ERROR]"

    async def _evaluate_batch(self, rows: List[dict]) -> List[dict]:
        # 이벤트 루프와 함께 사용할 semaphore를 여기서 생성
        semaphore = asyncio.Semaphore(5)  # 초당 최대 5병렬 요청
        async with httpx.AsyncClient() as client:
            tasks = [
                self._call_api(
                    self._make_prompt(row['err_sentence']),
                    client,
                    semaphore
                )
                for row in rows
            ]
            responses = await asyncio.gather(*tasks)
        return [
            {'id': row['id'], 'cor_sentence': res}
            for row, res in zip(rows, responses)
        ]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = data.to_dict(orient='records')
        results = asyncio.run(self._evaluate_batch(rows))
        return pd.DataFrame(results)

    def run_template_experiment(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame
    ) -> Dict:

        train_results = self.run(train_data)
        train_score = evaluate_correction(train_data, train_results)

        valid_results = self.run(valid_data)
        valid_score = evaluate_correction(valid_data, valid_results)

        return {
            'train_recall': train_score,
            'valid_recall': valid_score,
            'train_results': train_results,
            'valid_results': valid_results
        }