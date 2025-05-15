BASE_TEMPLATES = [
    {
        "id": "node_base_1",  # 간결형, 최소 지시
        "template": "다음 문장을 고쳐줘: {text}",
        "generation": 0,
        "parent_ids": []
    },
    {
        "id": "node_base_2",  # 세부 나열형
        "template": (
            "다음 문장의 맞춤법, 띄어쓰기, 문장 부호를 올바르게 수정하세요.\n"
            "문장: {text}"
        ),
        "generation": 0,
        "parent_ids": []
    },
    {
        "id": "node_base_3",  # 전문가 스타일
        "template": (
            "문법 전문가처럼 아래 문장을 교정해줘. 실수한 부분만 수정해줘.\n"
            "{text}"
        ),
        "generation": 0,
        "parent_ids": []
    },
    {
        "id": "node_base_4",  # 공손 지시형
        "template": (
            "오류가 포함된 문장을 올바른 형태로 고쳐주세요:\n"
            "{text}"
        ),
        "generation": 0,
        "parent_ids": []
    },
    {
        "id": "node_base_5",  # 강한 정밀 지시형
        "template": (
            "다음 문장을 문법적으로 완벽하게 교정해주세요. "
            "불필요한 수식은 하지 말고 정확하게 수정만 하세요.\n"
            "문장: {text}"
        ),
        "generation": 0,
        "parent_ids": []
    }
]
