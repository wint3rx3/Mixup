import os
from dotenv import load_dotenv
import json
from typing import List
import requests
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from openai import OpenAI

load_dotenv()

# Upstage API 설정
UPSTAGE_API_KEY = os.environ["UPSTAGE_API_KEY"]
SOLAR_EMBEDDING_URL = "https://api.upstage.ai/v1/embeddings"
SOLAR_LLM_MODEL = "solar-pro"

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1"
)

# Solar Embedding 래퍼
def get_solar_embedding(text: str) -> List[float]:
    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": "embedding-query"
    }
    response = requests.post(SOLAR_EMBEDDING_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

class SolarEmbeddingWrapper:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_solar_embedding(t) for t in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return get_solar_embedding(text)

# 벡터 DB 초기화
def init_vectorstore(persist_dir="rag"):
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print("🔄 벡터 DB 초기화 중...")
        loader = TextLoader("reference_docs/national_institute.txt", encoding="utf-8")
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        embedding = SolarEmbeddingWrapper()
        Chroma.from_documents(docs, embedding, persist_directory=persist_dir)
        print("✅ 벡터 DB 생성 완료")

# 전역 초기화
embedding = SolarEmbeddingWrapper()
vectorstore = Chroma(persist_directory="rag", embedding_function=embedding)
retriever = vectorstore.as_retriever()

# LLM 호출 함수
def call_solar_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=SOLAR_LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# context 검색
def retrieve_context(query: str, k: int = 3) -> str:
    docs = retriever.invoke(query)[:k]
    return "\n".join([doc.page_content for doc in docs])

def retrieve_error_patterns(k: int = None) -> str:
    path = "optimizer/prompt_error_patterns.jsonl"
    if not os.path.exists(path):
        return ""
    
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(l.strip()) for l in f if l.strip()]
    
    if not lines:
        return ""
    
    # 자동 결정
    if k is None:
        total = len(lines)
        k = min(10, max(3, total // 10))  # 예: 전체의 10% 또는 최대 10개

    samples = lines[-k:]
    formatted = []
    for err in samples:
        from_text = err.get("wrong_change_from", "").strip()
        to_text = err.get("wrong_change_to", "").strip()
        if from_text and to_text:
            formatted.append(f"- '{from_text}' → '{to_text}' 는 과잉 교정 가능성이 있으므로 피해야 함")
    return "\n".join(formatted)


def generate_prompts_batch(prompt_text: str, n: int = 3) -> List[str]:
    context = retrieve_context(prompt_text)
    error_hints = retrieve_error_patterns(k=10)

    batch_prompt = f"""
당신은 한국어 문법 교정 전문가이자 프롬프트 최적화 대회 참가자입니다.

아래의 문맥과 실패 사례를 참고하여 AI 모델이 사용할 프롬프트 {n}개를 생성하세요.

[목표]
- AI가 입력 문장의 문법, 맞춤법, 띄어쓰기, 조사, 어미, 문장 부호 오류를 정확하게 수정할 수 있도록 프롬프트를 설계하세요.
- 교정 결과는 정답 문장과 최대한 유사해야 하며, **과잉 수정은 반드시 피해야 합니다.**
- 교정 프롬프트는 실제 SNS, 커뮤니티 등 구어체 문장에도 잘 작동해야 하며, 실용적이고 구조적인 지시를 포함해야 합니다.

[작성 지침]
- 각 프롬프트는 반드시 {{text}} 자리표시자를 **정확히 1회만** 포함해야 합니다.
- 프롬프트는 1~5문장 사이로 자유롭게 작성하되, 문장 구조와 의도가 명확해야 합니다.
- 출력에는 교정된 문장 외에 설명, 분석, 마크다운이 포함되지 않도록 유도하세요.

[실패 교정 사례]
아래는 과거 프롬프트가 잘못된 교정을 유도한 예시입니다. 
이런 오류가 재발하지 않도록 설계에 반영하세요.

{error_hints}

[기존 템플릿]
{prompt_text}

[참고 문서 요약]
{context}

[개선된 프롬프트 목록]
"""
    response = call_solar_llm(batch_prompt)
    return [
        line.strip()
        for line in response.strip().split("\n")
        if "{text}" in line and line.strip().count("{text}") == 1 and len(line.strip()) > 10
    ][:n]

# 테스트
if __name__ == "__main__":
    init_vectorstore()
    seed_prompt = "다음 문장을 고쳐줘: {text}"

    print("\n 배치 개선 결과:")
    print(generate_prompts_batch(seed_prompt, n=3))