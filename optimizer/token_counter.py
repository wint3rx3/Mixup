# optimizer/token_counter.py

def count_tokens(text: str) -> int:
    """
    텍스트의 토큰 수를 간단히 계산합니다.
    실제로는 GPT 등의 토크나이저를 쓰는 것이 더 정확하지만,
    여기서는 공백 기준으로 분리한 수를 임시 토큰 수로 간주합니다.
    """
    if not text:
        return 0
    return len(text.strip().split())
