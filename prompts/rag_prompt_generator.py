# prompts/rag_prompt_generator.py

import os
import json
import random
from typing import Optional
from config import ERROR_PATTERN_PATH, DOCS_DIR

from engine.api_client import call_llm
from prompts.template_format import format_prompt
from prompts.prompt_validator import validate_template

PROMPT_ERROR_FILE = "data/prompt_error_patterns.jsonl"
DOCS_DIR = "docs/"

def load_error_context(n: int = 10, error_type_filter: Optional[str] = None):
    contexts = []
    if os.path.exists(PROMPT_ERROR_FILE):
        with open(PROMPT_ERROR_FILE, encoding="utf-8") as f:
            lines = list(f)
            random.shuffle(lines)  # ✅ 랜덤하게 섞기
            count = 0
            for line in lines:
                item = json.loads(line)
                if error_type_filter and item["error_type"] != error_type_filter:
                    continue
                contexts.append(f"- {item['error_type']}: \"{item['example']}\"")
                count += 1
                if count >= n:
                    break
    return "\n".join(contexts)

def load_docs_context() -> str:
    doc_texts = []
    if os.path.exists(DOCS_DIR):
        for fname in os.listdir(DOCS_DIR):
            if fname.endswith(".txt"):
                with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
                    doc_texts.append(f.read())
    return "\n".join(doc_texts)

async def generate_child_prompts(session, parent_nodes, n_children=3) -> list:
    all_children = []

    for parent in parent_nodes:
        example_errors = load_error_context(n=10)
        context = example_errors + "\n" + load_docs_context()

        # system + user 메시지 구성
        messages = [
            {
                "role": "system",
                "content": (
                    "너는 프롬프트 최적화를 전문으로 하는 AI 엔지니어야. "
                    "문법 교정 태스크에서 실제 사용 가능한 프롬프트 템플릿만 생성해야 해. "
                    "템플릿 형식은 부모와 동일한 single-turn 또는 multi-turn이어야 하며, "
                    "각 템플릿에는 반드시 `{text}` 자리표시자가 포함되어 있어야 해. "
                    "설명, 번호, 따옴표, 예시, 분석 문장은 절대 출력하지 마. "
                    "오직 템플릿 본문만 깔끔하게 출력해."
                )
            },
            {
                "role": "user",
                "content": f"""[문맥 요약]
        {context}

        [부모 템플릿]
        {json.dumps(parent.template, ensure_ascii=False, indent=2)}

        [실패 사례 요약]
        {example_errors}

        요청:
        - 부모 템플릿과 같은 목적(한국어 문장 교정)을 수행하지만, 표현 방식이나 문장 구조가 다른 템플릿을 {n_children}개 생성해줘.
        - 반드시 `{{text}}` 자리표시자가 포함되어야 하며, 실제 사용 가능한 템플릿이어야 해.
        - 아래 형식 지침을 반드시 지켜줘:

        [출력 형식 지침]
        - 각 템플릿은 한 줄로 출력
        - 설명, 번호, 따옴표, "→" 기호 등을 포함하지 마
        - 템플릿 본문 외의 어떤 문장도 출력하지 마
        """
            }
        ]

        response = await call_llm(messages)
        lines = [line.strip() for line in response.split("\n") if line.strip() and any(c.isalpha() for c in line)]

        for line in lines[:n_children]:
            cleaned = line  # 더 이상 ast.literal_eval 불필요

            child_template = {
                "template": cleaned,
                "type": parent.type,
                "description": f"Child of {parent.id[:8]}"
            }

            if validate_template(child_template["template"], verbose=True):
                all_children.append(child_template)
            else:
                print(f"❌ 무효 템플릿 (부모 {parent.id[:8]}): {cleaned[:60]}")

    return all_children
