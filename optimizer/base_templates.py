# optimizer/base_templates.py

BASE_TEMPLATES = [
    {
        "id": "node_base_1",
        "template": "다음 문장을 맞춤법과 문법에 맞게 수정해 주세요:\n{text}",
        "generation": 0,
        "parent_ids": []
    },
    {
        "id": "node_base_2",
        "template": (
            "아래 문장을 다음 항목에 따라 교정하세요.\n"
            "- 맞춤법\n"
            "- 띄어쓰기\n"
            "- 문법적 오류\n"
            "- 문장 부호\n\n"
            "문장: {text}"
        ),
        "generation": 0,
        "parent_ids": []
    },
    {
        "id": "node_base_3",
        "template": (
            "문장을 문법적으로 올바르게 다듬고, 필요 시 문장 부호와 띄어쓰기도 바로잡아 주세요.\n"
            "{text}"
        ),
        "generation": 0,
        "parent_ids": []
    },
    {
        "id": "node_base_4",
        "template": (
            "공식적인 글쓰기에 적합하도록 다음 문장을 정중하게 다듬어 주세요.\n"
            "- 존댓말 형태로 변환\n"
            "- 맞춤법, 띄어쓰기, 문장 부호 수정\n\n"
            "문장: {text}"
        ),
        "generation": 0,
        "parent_ids": []
    },
    {
        "id": "node_base_5",
        "template": (
            "다음 문장을 정확하고 자연스러운 한국어로 교정해 주세요. 문장 부호, 맞춤법, 띄어쓰기를 모두 고려하세요.\n"
            "문장: {text}"
        ),
        "generation": 0,
        "parent_ids": []
    }
]
