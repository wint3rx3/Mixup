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

load_dotenv()  # 반드시 호출해야 함

# ========= 🌞 Upstage API 설정 ========= #

UPSTAGE_API_KEY = os.environ["UPSTAGE_API_KEY"]
SOLAR_EMBEDDING_URL = "https://api.upstage.ai/v1/embeddings"
SOLAR_LLM_MODEL = "solar-pro"

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1"
)

# ========= 🌞 Solar Embedding 래퍼 ========= #

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

# ========= 🔧 벡터 DB 초기화 ========= #

def init_vectorstore(persist_dir="rag_db"):
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print("🔄 벡터 DB 초기화 중...")
        loader = TextLoader("reference_docs/national_institute.txt", encoding="utf-8")
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        embedding = SolarEmbeddingWrapper()
        Chroma.from_documents(docs, embedding, persist_directory=persist_dir)
        print("✅ 벡터 DB 생성 완료")

# ✅ 전역 초기화
embedding = SolarEmbeddingWrapper()
vectorstore = Chroma(persist_directory="rag_db", embedding_function=embedding)
retriever = vectorstore.as_retriever()

# ========= 🧠 LLM 호출 함수 ========= #

def call_solar_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=SOLAR_LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ========= 🔍 context 검색 ========= #

def retrieve_context(query: str, k: int = 3) -> str:
    docs = retriever.get_relevant_documents(query)[:k]
    return "\n".join([doc.page_content for doc in docs])

# ========= ✅ 단일 템플릿 개선 ========= #

def generate_prompt(prompt_text: str) -> str:
    context = retrieve_context(prompt_text)
    prompt_template = PromptTemplate(
        input_variables=["context", "text"],
        template="""다음은 한국어 맞춤법 교정 프롬프트입니다. 아래 참고 문서를 기반으로,
더 성능이 좋은 프롬프트를 생성해주세요.

[참고 문서 요약]
{context}

[기존 템플릿]
{text}

[개선된 템플릿]
"""
    )
    full_prompt = prompt_template.format(context=context, text=prompt_text)
    return call_solar_llm(full_prompt)

# ========= ✅ 여러 개 템플릿 생성 (batch) ========= #

def generate_prompts_batch(prompt_text: str, n: int = 3) -> List[str]:
    context = retrieve_context(prompt_text)
    batch_prompt = f"""
다음은 한국어 맞춤법 교정 프롬프트입니다. 아래 참고 문서를 기반으로,
새로운 교정 프롬프트 {n}개를 생성해주세요. 간결하고 명확하게 작성하세요.

[참고 문서 요약]
{context}

[기존 템플릿]
{prompt_text}

[개선된 템플릿 목록]
"""
    response = call_solar_llm(batch_prompt)
    return [line.strip() for line in response.strip().split("\n") if len(line.strip()) > 10][:n]

# ========= ✅ 테스트 ========= #

if __name__ == "__main__":
    init_vectorstore()

    seed_prompt = "다음 문장을 고쳐줘: {text}"

    print("\n✅ 단일 개선 결과:")
    print(generate_prompt(seed_prompt))

    print("\n✅ 배치 개선 결과:")
    print(generate_prompts_batch(seed_prompt, n=3))
