## 🔁 강화학습형 프롬프트 최적화

## 🗂️ 파일 디렉토리 구조
```text
grammar-correction-project/
│
├── .env                        # API 키 저장 (UPSTAGE_API_KEY)
├── requirements.txt
├── README.md                   # 전체 구조 및 실행 설명
│
├── data/                       # 원본 학습 및 평가 데이터
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── submission/                 # ✅ 최종 제출 전용 (룰에 맞춘 단일 템플릿 inference)
│   ├── main.py                 # test.csv → submission.csv 생성
│   ├── config.py
│   ├── prompts/templates.py    # 선정된 템플릿 1개
│   ├── utils/experiment.py
│   ├── utils/metrics.py
│   └── outputs/submission.csv
│
├── optimizer/                  # 🧪 프롬프트 최적화 및 강화 루프 전용
│
│   ├── optimize.py             # 전체 루프 제어: 실험 → 평가 → 개선
│
│   ├── prompts/                # 템플릿 저장소
│   │   ├── base_templates.py       # 수동 작성 초기 프롬프트
│   │   ├── rag_templates.py        # RAG 기반 개선 템플릿
│   │   ├── generated_templates.py  # 강화 루프 중 생성된 템플릿들
│   │   └── template_manager.py     # 템플릿 불러오기, 저장, 로깅
│
│   ├── utils/
│   │   ├── experiment.py       # 교정 실행 + 결과 수집
│   │   ├── metrics.py          # recall/precision 평가
│   │   ├── api_logger.py       # 문장별 호출 횟수 기록 + 3회 제한 검사
│   │   ├── token_counter.py    # 토큰 길이 검사 (2000 제한)
│   │   ├── prompt_validator.py # 리라이팅 위험, 금지 표현 필터링
│   │   ├── rag_utils.py        # 문법 문서 chunk 검색 + 인용
│   │   └── scoring.py          # 템플릿 점수 계산식
│
│   ├── reinforce/              # 🔁 프롬프트 진화 학습 루프
│   │   ├── reinforce_loop.py       # 반복 루프 (Top → 새 템플릿 N개 생성)
│   │   ├── template_rewriter.py    # LLM으로 프롬프트 생성 (Top K 기반)
│   │   └── memory.jsonl            # 세대별 템플릿 버전 기록
│
│   ├── rag/                    # 문법 기반 RAG 참조 자료
│   │   ├── grammar_reference.txt
│   │   └── grammar_chunks.jsonl
│
│   └── results/
│       ├── experiment_log.jsonl       # 실험별 성능 기록
│       ├── template_eval_scores.csv   # 템플릿 별 점수 비교표
│       └── risky_templates.csv        # 필터링 대상 로그
```

```
grammar-correction-project/
│
├── .env                        # API 키 저장 (UPSTAGE_API_KEY)
├── requirements.txt
├── README.md                   # 전체 구조 및 실행 설명
│
├── data/                       # 원본 학습 및 평가 데이터
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── submission/                 # ✅ 최종 제출 전용 (룰에 맞춘 단일 템플릿 inference)
│   ├── main.py                 # test.csv → submission.csv 생성
│   ├── config.py
│   ├── prompts/templates.py    # 선정된 템플릿 1개
│   ├── utils/experiment.py
│   ├── utils/metrics.py
│   └── outputs/submission.csv
│
├── optimizer/                  # 🧪 프롬프트 최적화 및 강화 루프 전용
│   ├── optimize.py                 # 🔁 전체 강화 루프 제어 (세대 반복)
│   ├── async_runner.py             # ⚡ 비동기 평가기 (batch 평가)
│   ├── template_rewriter.py        # 🧠 LLM 기반 자식 템플릿 생성기
│   ├── reinforce_graph.py          # 🌳 템플릿 그래프 관리 + memory 기록
│   ├── prompt_validator.py         # 🛡️ 위험 프롬프트 필터링 (금지 표현, 등)
│   ├── token_counter.py            # 🔢 토큰 길이 검사 (2000 제한)
│   └── memory.jsonl                # 🧠 템플릿 세대, 성능 기록 (그래프)
│
├── rag/                        # 📚 문법 기반 RAG 참조 자료 (선택)
│   ├── grammar_reference.txt
│   └── grammar_chunks.jsonl
│
└── utils/                      # 공통 유틸리티
    ├── experiment.py               # 기존 동기 평가기 (baseline 용)
    ├── metrics.py                  # recall/precision 측정

```