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
                    "너는 한국어 문법 교정 AI의 프롬프트를 생성하는 전문 엔지니어야.\n"
                    "우리는 이 템플릿으로 문장에서 오탈자, 띄어쓰기, 문장부호 오류를 고치게 만들 것이며,\n"
                    "**최종 출력이 정확한 1줄의 문장만 나오도록 템플릿을 설계**해야 해.\n\n"

                    "이 대회에서 가장 중요한 것은:\n"
                    "1. 문장의 의미, 말투, 어조를 바꾸지 않고\n"
                    "2. 실제로 틀린 것만 고치며\n"
                    "3. **추론 결과로 반드시 하나의 문장만 출력되도록** 제어하는 것이야.\n\n"

                    "템플릿은 반드시 `{{text}}` 자리표시자를 포함해야 하고,\n"
                    "**설명, 예시, 번호, 따옴표, '정답은' 등의 불필요한 문구는 절대 포함하면 안 돼.**\n"
                    "**멀티턴 템플릿에서는 'system'과 'user'만 사용할 수 있고, 'assistant'는 포함하지 마.**\n"
                    "**LLM이 문장을 하나만 출력하도록 유도하는 것이 핵심**이며, 문장 외 텍스트가 섞이면 평가에서 탈락하게 돼.\n"
                )
            },
            {
                "role": "user",
                "content": f"""
        ✅ 아래 정보를 모두 반영해서 `{parent.type}` 유형의 템플릿을 {n_children}개 생성해줘.

        ---

        ## 1. 최근 실패 요약 (이런 오류는 다시 발생하지 않게 막아야 해)
        - 마침표 누락으로 문장이 비완성 처리됨
        - 문장에 쉼표나 분석 설명이 출력되어 평가에서 제외됨
        - ‘문제풀이’를 ‘문제 풀이’로 바꿔 오답 처리됨
        - ‘맞는 듯 해’ 대신 ‘맞는듯해’를 사용하지 않음
        - 정답 문장 앞에 ‘정답:’, ‘→’ 등이 출력되어 채점 실패

        ➡ 위와 같은 오류가 발생하지 않도록 템플릿 내부에서 **명시적으로 제약 조건을 설정**해야 해

        ---

        ## 2. 교정 기준 (CHECK_SINGLE + train 문서 기반)

        ### ① 오탈자 (철자 오류)
        - 자판 실수: '됬다' → '됐다', '잇서요' → '있어요'
        - 발음 기반 표기: '머르겠서요' → '모르겠어요', '믈론' → '물론'
        - 잘못된 외래어 표기: '썸네일' → '섬네일'
        - 존재하지 않는 단어: '읃따' → '얻다'

        ### ② 띄어쓰기
        - 보조 용언/조사: '할수있다' → '할 수 있다'
        - 반드시 **띄어 써야 하는 표현**:
        - '샤프 심', '교직 이수', '전공 수업', '후 순위', '조회 수'
        - 반드시 **붙여 써야 하는 표현**:
        - '시험기간', '정정기간', '한번', '필기노트', '공부시간'
        - 관용 표현은 붙여 써야 함: '도와주셔서', '맞는듯해', '해주셨다', '터득해가지고'

        ### ③ 문장부호
        - 문장 끝에 종결부호가 없으면 마침표(.) 또는 물음표(?) 추가
        - 쉼표(,)는 원문에 있을 때만 유지. 새로 추가하지 말 것
        - 생략부호(...), 느낌표(!), 이모티콘 등은 고치지 말고 유지

        ---

        ## 3. 금지 규칙 (아주 중요!)

        - 어조, 말투, 문체, 높임법은 **절대 바꾸지 마**
        - '거' → '것', '있을까요' → '있으실까요' 금지
        - 같은 의미라도 표현 바꾸지 마: '도와주세요' → '도움 주세요' ❌
        - 조사/어미 임의 추가 금지: '맞는지' → '맞는지가' ❌
        - 설명, 분석, 예시 출력 금지: '정답은', '→', '수정된 문장:' 등 **절대 금지**
        - ‘문제풀이’와 ‘문제 풀이’는 둘 다 정답 → 이미 맞는 표현은 고치지 마

        ---

        ## 4. 출력 형식 제약

        - 출력은 **문장 하나**만, 마침표/물음표/느낌표로 끝나는 문장만 출력
        - 문장 외 텍스트, 설명, 따옴표, 줄바꿈, 이모지, 해설 등은 **모두 제거**
        - JSON 저장 시 `"cor_sentence"` 필드로 바로 들어갈 수 있어야 함
        - 프롬프트 길이는 2000자 이내여야 함 (너무 짧지도, 너무 단순하지도 않게)
        - 설명, 번호, 따옴표, 예시, 마크다운 포맷(예: ###, ```markdown, Template 1:) 등은 절대 포함하지 마.
        - 오직 템플릿 문장 본문만 출력하고, 번호나 주석 없이 한 줄로 출력해.

        ---

        ## 5. 생성 목표

        - {n_children}개의 `{parent.type}` 템플릿을 생성해줘
        - 프롬프트 본문만 출력하고, 각각 서로 구조가 다르게 구성되도록 표현 다양성 확보해
        - LLM이 잘못된 문장을 출력하지 않도록 철저한 통제를 적용해
        - 가능한 한 상세하게, 가능한 한 많은 규칙과 제약을 포함해줘

        🎯 이 템플릿으로 우리는 문법 교정 대회에서 1등을 해야 해.  
        실패 예시를 되풀이하지 않도록, 최고의 프롬프트를 만들어줘.
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
