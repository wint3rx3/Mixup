# config.py

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 API 키 로딩

# 📁 경로 설정
GRAPH_PATH = "data/prompt_graph.jsonl"
RESULTS_PATH = "data/results.jsonl"
MEMORY_PATH = "data/memory.jsonl"
ERROR_PATTERN_PATH = "data/prompt_error_patterns.jsonl"
DOCS_DIR = "docs/"

# 📊 평가 기준
RECALL_THRESHOLD = 0.6

# 🧠 템플릿 검증
MAX_CHAR_LENGTH = 2000
FORBIDDEN_PHRASES = [
    "정답은", "수정된 문장", "→", "다음은", "수동", "참고자료", "위키백과", "설명드리자면"
]
ALLOWED_ROLES = {"system", "user"}

# 🧪 실험 파라미터
SAMPLE_SIZE = 750
TOP_K = 3
N_CHILDREN = 2
