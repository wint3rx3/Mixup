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
                    "너는 한국어 문법 교정 AI의 프롬프트를 생성하는 최고 전문가야.\n"
                    "우리는 이 템플릿으로 문장에서 오탈자, 띄어쓰기, 문장부호 오류만 고치게 만들 것이며,\n"
                    "**추론 결과로 반드시 문장 1줄만 출력되도록 템플릿을 설계해야 해.**\n\n"

                    "프롬프트는 반드시 `{{text}}` 자리표시자를 포함해야 해.\n"
                    "**설명, 예시, 번호, 따옴표, 마크다운 등은 절대 포함하지 마.**\n"
                    "**템플릿 본문만 한 줄로 출력해야 하며**, 줄바꿈, 분석 설명, 출력 지시어도 안 돼.\n"
                    "**멀티턴 템플릿은 'system'과 'user'만 사용할 수 있고, 'assistant'는 사용하지 마.**"
                )
            },
            {
                "role": "user",
                "content": f"""
        🎯 아래 조건을 충족하는 `{parent.type}` 유형의 템플릿을 총 {n_children}개 생성해줘.  
        **한 줄짜리 템플릿 문장만** 출력해.  
        **번호, 따옴표, 마크다운, 설명 없이** 템플릿 본문만 나열해야 해.  
        각 템플릿은 반드시 `{{text}}`를 포함해야 하며, 출력은 아래 예시처럼 간결하고 정제된 문장 형태여야 해.

        ---

        📌 **실패 예시 요약**
        - 마침표 누락으로 문장이 비완성 처리됨
        - 문장에 쉼표나 분석 설명이 출력되어 평가에서 제외됨
        - ‘문제풀이’를 ‘문제 풀이’로 바꿔 오답 처리됨
        - ‘맞는 듯 해’ 대신 ‘맞는듯해’를 사용하지 않음
        - 정답 문장 앞에 ‘정답:’, ‘→’, '수정된 문장:' 등의 출력으로 채점 실패

        ➡ 위와 같은 오류를 반복하지 않도록 템플릿 안에 **명확한 제약 조건**을 포함해줘.

        ---

        📌 **교정 기준 요약**

        ① 오탈자 (철자 오류)
        - '됬다' → '됐다', '믈론' → '물론', '읃따' → '얻다'

        ② 띄어쓰기
        - 반드시 띄어 써야 함: '샤프 심', '교직 이수', '조회 수'
        - 반드시 붙여 써야 함: '시험기간', '필기노트', '공부시간'
        - 관용 표현은 붙여 써야 함: '도와주셔서', '맞는듯해', '터득해가지고'

        ③ 문장부호
        - 문장 끝에 마침표/물음표/느낌표가 없으면 추가
        - 쉼표는 원문에 있을 때만 유지 (새로 추가 금지)
        - 생략부호(...)나 느낌표(!) 등은 고치지 말 것

        ---

        📌 **금지 사항**
        - 문체, 말투, 표현 방식 바꾸기 금지 ('거' → '것', '있을까요' → '있으실까요' 등 ❌)
        - 설명/해설/예시 출력 금지 ('정답은', '→', 'Template 1:', 따옴표 등 ❌)
        - 문장이 아닌 요소 출력 금지 (분석, 요약, 마크다운 ❌)
        - 종결부호 없이 끝나는 문장 ❌

        ---

        📌 **출력 예시 (형식 참고만, 그대로 출력하지 마):**

        - 문장에서 오탈자와 띄어쓰기, 문장부호 오류만 고쳐서 한 문장으로 출력해. 입력: {{text}}
        - 다음 문장에서 문법 오류만 고쳐. {{text}}

        ---

        이제 위 조건을 모두 반영해서 `{n_children}`개의 `{parent.type}` 템플릿을 **아래와 같이 한 줄씩 출력해줘.**

        - ✅ 각 줄은 템플릿 본문 1개
        - ✅ 각 줄은 반드시 `{{text}}`를 포함
        - ✅ 따옴표 없이, 번호 없이, 마크다운 없이
        - ✅ 문장의 의미나 말투는 그대로 유지하되, 문법 오류만 고치게 유도
        - ✅ 출력은 오직 문장 1줄만 하게 유도
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
