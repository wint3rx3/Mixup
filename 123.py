# env_test.py
import os
from dotenv import load_dotenv

load_dotenv()  # 반드시 호출해야 함

print("📦 현재 디렉토리:", os.getcwd())
print("🔍 .env 내 UPSTAGE_API_KEY:", os.getenv("UPSTAGE_API_KEY"))
