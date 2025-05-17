# prompts/prompt_validator.py

from config import MAX_CHAR_LENGTH, FORBIDDEN_PHRASES, ALLOWED_ROLES
from typing import Union, List, Dict
import re

FORBIDDEN_PHRASES = [
    "정답은", "수정된 문장", "→", "다음은", "수동", "참고자료", "위키백과", "설명드리자면"
]
ALLOWED_ROLES = {"system", "user"}

def extract_prompt_string(template: Union[str, List[Dict]]) -> str:
    if isinstance(template, str):
        return template
    elif isinstance(template, list):
        return " ".join(turn.get("content", "") for turn in template)
    return ""

def validate_template(template: Union[str, List[Dict]], verbose: bool = False) -> bool:
    prompt_str = extract_prompt_string(template)

    if len(prompt_str) > MAX_CHAR_LENGTH:
        if verbose:
            print("❌ [길이 초과] 템플릿이 2000자를 초과했습니다.")
        return False

    if any(phrase in prompt_str for phrase in FORBIDDEN_PHRASES):
        if verbose:
            print("❌ [금지 표현] 템플릿에 금지된 문구가 포함되어 있습니다.")
        return False

    if isinstance(template, list):
        roles = [t.get("role") for t in template]
        if not set(roles).issubset(ALLOWED_ROLES):
            if verbose:
                print(f"❌ [역할 오류] 허용되지 않은 role이 포함됨: {set(roles) - ALLOWED_ROLES}")
            return False
        if "user" not in roles:
            if verbose:
                print("❌ [구성 오류] user 역할이 포함되어 있지 않습니다.")
            return False
        if not any("{text}" in t.get("content", "") for t in template):
            if verbose:
                print("❌ [자리표시자 누락] 멀티턴 템플릿에 {{text}}가 포함되어 있지 않습니다.")
            return False
    else:
        if "{text}" not in prompt_str:
            if verbose:
                print("❌ [자리표시자 누락] 템플릿에 {{text}}가 포함되어 있지 않습니다.")
            return False

    return True

