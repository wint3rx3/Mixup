# improved_templates.py

IMPROVED_TEMPLATES = [
    {
        "id": "improved_01",
        "type": "multi-turn",
        "template": [
            {
                "role": "system",
                "content": (
                    "너는 한국어 문장의 문법 오류를 교정하는 고성능 교정기야. 문맥에 맞게 철자, 띄어쓰기, 종결 부호만 수정해야 해.\n\n"
                    "단어 자체의 의미, 말투, 어투, 조사 추가 등은 금지야. 조사/어미 변경 없이 문맥상 맞는 경우에만 고쳐야 해.\n\n"
                    "출력은 반드시 교정된 문장 한 줄만, 마침표/물음표/느낌표로 끝나게 해."
                )
            },
            {
                "role": "user",
                "content": "입력문장: {text}"
            }
        ],
        "description": "best_single_01 기반 간결 템플릿"
    },
    {
        "id": "improved_02",
        "type": "single-turn",
        "template": (
            "다음 문장에서 오탈자, 띄어쓰기, 문장부호 오류만 고쳐줘. 의미, 문체, 말투는 그대로 유지하고, 출력은 한 줄로만 해.\n\n"
            "입력문장: {text}"
        ),
        "description": "간단한 단일턴 템플릿"
    }
]
