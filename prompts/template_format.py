# prompts/template_format.py

from typing import Union, List, Dict

def format_prompt(template: Union[str, List[Dict]], text: str) -> List[Dict]:
    """
    템플릿(str 또는 multi-turn list)을 LLM 호출에 맞게 message 포맷으로 변환
    """
    if isinstance(template, str):  # 싱글턴
        prompt = template.replace("{text}", text)
        return [{"role": "user", "content": prompt}]
    
    elif isinstance(template, list):  # 멀티턴
        formatted = []
        for turn in template:
            formatted.append({
                "role": turn["role"],
                "content": turn["content"].replace("{text}", text)
            })
        return formatted
    
    else:
        raise ValueError("템플릿 형식이 올바르지 않습니다.")
