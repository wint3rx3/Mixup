import re

# 금지 표현: 사용자 행동 유도, 설명형 문장
FORBIDDEN_PHRASES = [
    "직접 수정해보세요",
    "예시는 다음과 같습니다",
    "문장을 분석하면",
    "설명드리겠습니다",
    "이제 고쳐볼게요",
    "교정 결과입니다",
    "정답은 다음과 같습니다",
    "사용자께서 확인해주세요"
]

# 위험 패턴: 설명, 반복 지시, 역할 변환, 과도한 출력 마커
RISKY_PATTERNS = [
    r"다음과 같은.*결과",
    r"~할 수 있습니다",
    r"제 생각에는",
    r"정답[:：]",
    r"출력 결과",
    r"결과는",
    r"{text}.*{text}",
    r"[^\s]{200,}",  # 너무 긴 텍스트 블록
]

# 토큰 수 계산 (간단한 whitespace 기반)
def count_tokens(text: str) -> int:
    return len(text.strip().split())

# 개별 검사
def is_forbidden(template: str) -> bool:
    return any(phrase in template for phrase in FORBIDDEN_PHRASES)

def is_risky(template: str) -> bool:
    return any(re.search(pattern, template) for pattern in RISKY_PATTERNS)

# 최종 유효성 판단
def validate_template(template: str, verbose: bool = False) -> bool:
    if is_forbidden(template):
        if verbose: print("❌ 금지 표현 포함")
        return False
    if is_risky(template):
        if verbose: print("❌ 위험 패턴 포함")
        return False
    if template.count("{text}") != 1:
        if verbose: print("❌ {text} 자리표시자 중복 또는 없음")
        return False
    if count_tokens(template) > 2000:
        if verbose: print(f"❌ 토큰 수 초과: {count_tokens(template)} tokens")
        return False
    return True
