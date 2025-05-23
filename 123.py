import os
import pandas as pd
import asyncio
from engine.api_client import call_llm
from tqdm.asyncio import tqdm_asyncio
from optimizer.metrics import evaluate_correction
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from openai import OpenAI

def get_upstage_token_count(texts: list[str], api_key: str, model: str = "embedding-passage") -> int:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1"
    )

    response = client.embeddings.create(
        model=model,
        input=texts
    )

    return response.usage.prompt_tokens  # ✅ 여기 수정됨


# 템플릿 적용 (Multi-turn)
def apply_template(template_name: str, text: str) -> List[Dict]:
    rule_text = (
        "너는 한국어 문장의 오탈자, 띄어쓰기, 문장부호 오류를 정밀하게 교정하는 전문가야. "
        "입력 문장의 의미, 어투, 말투는 **절대 바꾸지 말고**, 오직 잘못된 부분만 수정해야 해.\n\n"

        "📌 **교정 원칙**\n"
        "1. **오탈자(철자 오류)**\n"
        "- 자판 실수, 철자 오류, 자음/모음 혼동, 유사한 철자나 발음으로 인한 혼동을 고쳐줘. 예: '됬다' → '됐다'\n"
        "- 존재하지 않거나 의도가 불분명한 단어는 문맥상 자연스럽고 정확한 단어로 수정해줘.\n"
        "- 단어는 존재하지만 문맥에 부자연스러운 경우, 가장 적절한 단어로 바꿔줘. 단, 의미가 유지되어야 해.\n\n"

        "2. **띄어쓰기**\n"
        "- 띄어쓰기는 의미 단위와 문법 단위를 기준으로 교정해야 해.\n"
        "- 다음 단어는 반드시 **띄어 써야 해**:\n"
        "  · '조회 수', '전공 수업', '행동 강령', '수능 문학', '샤프 심', '백지 복습', '후 순위', '교직 이수', '배면 발광',\n"
        "    '전면 발광', '지역 페이', '초등 교사', '가짜 뉴스', '자음군 단순화', '기숙 학원', '만국 공법', '공부 의욕', '노인 성비',\n"
        "    '극 초반', '연극 영화과'\n"
        "- 다음 단어는 반드시 **붙여 써야 해**:\n"
        "  · '수시접수', '시험기간', '구조독해', '동해바다', '학교생활', '공부시간', '정정기간', '인구수', '의사분', '회전주기',\n"
        "    '공부자극', '악성코드', '나비효과', '필기노트', '주격조사', '수학학원', '인지심리학', '대학과정', '편가르기', '학습방향',\n"
        "    '치고는요', '교수되신', '회전관성', '파생접사', '과잠사진', '사형제도', '통합검색'\n"
        "- 관용 표현은 반드시 붙여 써. 예: '해주셨다', '도와주셔서', '한밤중'\n"
        "- ‘듯’, ‘해가지고’, ‘터득해가지고’ 같은 표현은 무조건 붙여 써야 해.\n\n"

        "3. **문장부호**\n"
        "- 문장 끝에 종결 부호가 없으면 문장 유형에 따라 마침표(.), 물음표(?), 느낌표(!) 중 하나를 추가해줘.\n"
        "- 쉼표(,)는 원문에 있을 때만 유지하고, 새로 삽입하지 마.\n"
        "- 생략부호(...), 복합부호(...?, ..!, ?!, 등)은 원형 그대로 유지하고 수정하지 마.\n"
        "- 원문에 없는 따옴표, 괄호, 특수부호는 삽입하지 마.\n\n"

        "📌 **절대 하지 마**\n"
        "- 말투나 격식 변경 금지: '선생' → '선생님', '거' → '것' ❌\n"
        "- 의미가 같다고 단어 자체를 바꾸면 안 돼: '있을까요' → '있으실까요', '밑에' → '아래' ❌\n"
        "- 조사 임의 추가 금지. 단, 오류는 고쳐도 돼. (예: 이, 가, 을, 를, 은, 는, 도, 만, 까지 등)\n"
        "- 문장이 어색해도 의미가 통하면 구조를 바꾸지 마.\n"
        "- 쉼표나 따옴표는 원문에 없으면 절대 삽입하지 마.\n\n"

        "📌 **반드시 다음과 같이 교정해야 해**\n"
        "- '그치만' → '그렇지만'\n"
        "- '스험생활' → '수험생활'\n"
        "- '썸네일' → '섬네일'\n"
        "- '평행 이동' → '평행이동'\n"
        "- '바뀌어 가지고' → '바뀌어가지고'\n"
        "- '터득 해가지고' → '터득해가지고'\n"
        "- '맞는 듯 해' → '맞는듯해'\n"
        "- '없는 듯' → '없는듯'\n"
        "- '해 주셨다' → '해주셨다'\n"
        "- '도와 주셔서' → '도와주셔서'\n"
        "- '쪼끔'은 '쪼금'으로, '조금'은 '조금' 그대로 유지\n"
        "- '문제풀이'와 '문제 풀이'는 둘 다 맞음 → 절대 수정하지 마\n"
        "- '트린' → '틀린'\n"
        "- '멘날' → '맨날'\n"
        "- '숨참구' → '숨 참고'\n"
        "- '한거음 한거음' → '한걸음 한걸음'\n"
        "- '재심장' → '제 심장'\n"
        "- '카페인음료순' → '카페인 음료수는'\n"
        "- '고치구나서' → '고치고 나서'\n"
        "- '거리면 괜찮다' → '걸리면 괜찮다'\n"
        "- '이해못할' → '미해 못 할'\n"
        "- '받았엇어' → '받았었어'\n"
        "- '공매도는다있잖아' → '공매도는 다 있잖아'\n"
        "- '응원해주셔서' → '응원해 주셔서'\n"
        "- '받아 드릴' → '받아드릴'\n"
        "- '아니구성적' → '아니고 성적'\n"
        "- '전문항' → '전 문항'\n"
        "- '존재하는거라' → '존재하는 거라'\n"
        "- '멀' → '뭘'\n"
        "- '점' → '좀'\n"
        "- '관련으루다가요' → '관련으로다가요'\n"
        "- '지금은안되요' → '지금은 안돼요'\n"
        "- '어떡해설쩡' → '어떻게 설정'\n"
        "- '학교급간' → '학교급 간'\n"
        "- '가돼면' → '가 되면'\n"
        "- '같이나가시는데' → '같이 나가시는데'\n"
        "- '감성있다고 한' → '감성 있다고 한'\n"
        "- '걸리는건가요' → '걸리는 건가요'\n"
        "- '고전적이미에서' → '고전적 의미에서'\n"
        "- '공부할때의' → '공부할 때의'\n"
        "- '공매도는다있잖아.' → '공매도는 다 있잖아.'\n"
        "- '그리구간격' → '그리고 간격'\n"
        "- '그말들어보니까' → '그 말 들어보니까'\n"
        "- '기대돼서요.' → '기대돼서요'\n"
        "- '느낌이달라지더라고요.' → '느낌이 달라지더라고요.'\n"
        "- '멘탈이약해졌어.' → '멘탈이 약해졌어.'\n"
        "- '밑에서윗사람' → '밑에서 윗사람'\n"
        "- '뭔가부족하다' → '뭔가 부족하다'\n"
        "- '받았엇어' → '받았었어'\n"
        "- '살짝무서운것같아요' → '살짝 무서운 것 같아요'\n"
        "- '생각안나서요' → '생각 안 나서요'\n"
        "- '성장해가고있는' → '성장해 가고 있는'\n"
        "- '숨참구' → '숨 참고'\n"
        "- '시간 깨 줄어들어요' → '시간 꽤 줄어들어요'\n"
        "- '신뢰할만한정보인가요?' → '신뢰할 만한 정보인가요?'\n"
        "- '아니구성적' → '아니고 성적'\n"
        "- '안되구' → '안 되고'\n"
        "- '어디가지' → '어디 가지'\n"
        "- '어떡해설쩡' → '어떻게 설정'\n"
        "- '없어진거같아요.' → '없어진 거 같아요.'\n"
        "- '왜죠?' → '왜죠?'\n"
        "- '이거이' → '이것이'\n"
        "- '이해못할' → '미해 못 할'\n"
        "- '자기자신을믿으세요' → '자기 자신을 믿으세요'\n"
        "- '정말이쁜것같아요' → '정말 예쁜 것 같아요'\n"
        "- '좋은하루되세요' → '좋은 하루 되세요'\n"
        "- '지금은안되요' → '지금은 안돼요'\n"
        "- '지나가버리더라고요.' → '지나가 버리더라고요.'\n"
        "- '책보니까' → '책 보니까'\n"
        "- '카페인음료순' → '카페인 음료수는'\n"
        "- '표현한거잖아요' → '표현한 거잖아요'\n"
        "- '하루하루가' → '하루하루가'\n"
        "- '한거음 한거음' → '한 걸음 한 걸음'\n"
        "- '한번에해결' → '한 번에 해결'\n"
        "- '해주셔서' → '해 주셔서'\n"
        "- '몆시간 하셧' → '는 몇 시간 하셨'\n"
        "- '밖에 없겠죵' → '밖에 없겠죠.'\n"
        "- '하루 잘해' → '하루 잘 해 '\n"
        "- '가르쳐 주새' → '가르쳐 주세'\n"
        "- '가서 해야겟다' → '가서 해야겠다.'\n"
        "- '강좌는 언재' → '강좌는 언제'\n"
        "- '걍 그대로일거' → '그냥 그대로일 거 '\n"
        "- '걍 합성해두' → '그냥 합성해도'\n"
        "- '거' → '거 '\n"
        "- '거 가타' → '거 같아'\n"
        "- '거 가타요' → '거 같아요.'\n"
        "- '거도하기싫은' → '것도 하기 싫은 '\n"
        "- '거면 어쩔수' → '거면 어쩔 수 '\n"
        "- '거요' → '거요.'\n"
        "- '거이 맞는 설명' → '것이 맞는 설명 '\n"
        "- '건 사실인데오' → '건 사실인데요'\n"
        "- '것은 맞지안' → '것은 맞지 않'\n"
        "- '게 아니죠' → '게 아니죠.'\n"
        "- '고런지 설명해' → '그런지 설명해 '\n"
        "- '구' → '고'\n"
        "- ' 가고싶네요' → '을 가고 싶네요.'\n"
        "- ' 내의 체대입시' → '의 체대 입시를'\n"
        "- ' 아니거든요' → '이 아니거든요.'\n"
        "- ' 하나 볼떄' → '를 하나 볼 때'\n"
        "- ' 할떄' → '할 때'\n"
        "- '가족이생기니 조' → ' 가족이 생기니 좋'\n"
        "- '가튼 족인' → ' 같은 족인 '\n"
        "- '갖고 가지 앉' → ' 갖고 가지 않'\n"
        "- '개라고 봐야' → ' 개라고 봐야 '\n"
        "- '개를 연습해' → ' 개를 연습해 '\n"
        "- '걍 궁금해서근' → '그냥 궁금해서 그런'\n"
        "- '걔' → '계'\n"
        "- '거' → ' 거 '\n"
        "- '거' → '것'\n"
        "- '거 가타' → ' 거 같아'\n"
        "- '거 가튼데 외' → ' 것 같은데 왜'\n"
        "- '거 갇' → ' 거 같'\n"
        "- '거 마늠' → ' 거 많음.'\n"
        "- '거어떄' → ' 거 어때?'\n"
        "- '거에요' → ' 거예요.'\n"
        "- '건 잘못됀' → ' 건 잘못된'\n"
        "- '건가요' → ' 건가요?'\n"
        "- '걸 안조' → ' 걸 안 좋'\n"
        "- '겁니다' → ' 겁니다.'\n"
        "- '것같습니다' → ' 것 같습니다.'\n"
        "- '겅부가 남았다' → '공부가 남았다.'\n"
        "- '게 좋거든요' → ' 게 좋거든요!'\n"
        "- '겟내요' → '겠네요.'\n"
        "- '과정은 외 가치' → ' 과정은 왜 가치 '\n"
        "- '구' → '고'\n"
        "- '구 묻는' → '고 묻는 '\n"
        "- '구 할수잇' → '고 할 수 있'\n"
        "- '구 현명한' → '고 현명한 '\n"
        "- '구요' → '고요.'\n"
        "- '구전에서 적으로' → '고전에서 적으로 '\n"
        "- '구해도 틀린거' → '고 해도 틀린 것'\n"
        "- '그거을 참았다는' → ' 그것을 참았다는 '\n"
        "- '글고 재' → '그리고 제'\n"
        "- '남았다고 체감해따' → ' 남았다고 체감했다'\n"
        "- '너머ㅓ갈줄 알아야' → '넘어갈 줄 알아야 '\n"
        "- '넒어지는법이 잇쓸' → '넓어지는 법이 있을'\n"
        "- '넘무 클거같애' → ' 너무 클 것 같아'\n"
        "- '대' → '데'\n"
        "- '대 무엇인지 궁금함' → '데 무엇인지 궁금합'\n"
        "- '대 잘못' → '데 잘 못 '\n"
        "- '대외 중력이' → '데 왜 중력이 '\n"
        "- '됀다' → ' 된다.'\n"
        "- '되니까요' → ' 되니까요.'\n"
        "- '드를까하늕데 점' → '들을까 하는데 좀'\n"
        "- '떄 티비' → '때 티브이 '\n"
        "- '떻하죠' → '떡하죠?'\n"
        "- '많아서 고생을 햇' → ' 많아서 고생을 했'\n"
        "- '말구는 미적븐' → ' 말고는 미적분'\n"
        "- '말해 상태가 갔' → ' 말해 상태가 같'\n"
        "- '문제 고칠떄' → ' 문제 고칠 때'\n"
        "- '므 어려우면 그런' → '무 어려우면 그런 '\n"
        "- '받아서 들어보시는' → '해서 들어보시는 '\n"
        "- '방향도 궁금하내' → ' 방향도 궁금하네'\n"
        "- '보면 쫌' → ' 보면 좀'\n"
        "- '보이구요' → ' 보이고요.'\n"
        "- '봄니다' → '봅니다.'\n"
        "- '봄미다' → ' 봄니다.'\n"
        "- '붙구 바로 군대' → ' 붙고 바로 군대 '\n"
        "- '사진 첨부햇' → ' 사진 첨부했'\n"
        "- '서버 생각나니' → ' 서버 생각나네'\n"
        "- '수는 없는거' → ' 수는 없는 것'\n"
        "- '스' → ' 수 '\n"
        "- '스스' → '할 수 '\n"
        "- '안되' → ' 안 돼'\n"
        "- '않겠지만 감사함' → ' 않겠지만 감사합'\n"
        "- '앵간하게 풉니다' → '엔간하게 풉니다.'\n"
        "- '얘전에잇던 일' → ' 예전에 있던 일 '\n"
        "- '어떡게 알수잇' → ' 어떻게 알 수 있'\n"
        "- '어떡케 머리' → ' 어떻게 머릿'\n"
        "- '엄에서는 어떠케' → '험에서는 어떻게'\n"
        "- '에' → '예'\n"
        "- '외 농노를' → ' 왜 농노를 '\n"
        "- '외 집단이' → '왜 집단이 '\n"
        "- '으로 푸는' → '로 푸는 '\n"
        "- '이게 끗인가요' → ' 이게 끝인가요?'\n"
        "- '이는 왤케' → ' 이는 왜 '\n"
        "- '이유를 알구싶씁' → ' 이유를 알고 싶습'\n"
        "- '잇는데, 이' → ' 있는데, 이 '\n"
        "- '자는거 가타요' → ' 자는 것 같아요.'\n"
        "- '잔 안은문장 둘' → '장, 안은문장 둘 '\n"
        "- '재 개강해서 언재' → '제 개강해서 언제'\n"
        "- '재쯤 나오나요' → '제쯤 나오나요?'\n"
        "- '지금시간이' → ' 지금 시간이 '\n"
        "- '틀리구' → ' 틀리고'\n"
        "- '풀었던 학생임' → ' 풀었던 학생입'\n"
        "- '할거같아요' → ' 할 거 같아요.'\n"
        "- '힘든지경이' → ' 힘든 지경이 '\n"

        "- 이 외에도 틀린 예시가 나오면 반드시 문맥에 맞게 고쳐줘.\n\n"

        "📌 **절대 고치지 말고 그대로 출력해야 하는 표현들**\n"
        "- '바랬다', '계속하셔요', '좋은건', '가득가득', '맞춰보는', '세부정보', '밑줄하고', '거임', '몇인지', '잘하면', '맞다면'\n\n"

        "📌 **출력 규칙**\n"
        "- 반드시 **교정된 문장 한 줄만** 출력해.\n"
        "- '결과:', '수정된 문장:' 같은 접두어나 설명은 쓰지 마.\n"
        "- 줄바꿈 없이 끝나야 하고, 반드시 종결부호(., ?, !)로 끝나야 해.\n"
    )

    return [
        {"role": "system", "content": rule_text},
        {"role": "user", "content": f"다음 문장을 교정해줘:\n{text}"}
    ]

# 비동기 LLM 호출
async def correct_row(template_name: str, text: str) -> str:
    messages = apply_template(template_name, text)
    return await call_llm(messages)

async def run_all(df: pd.DataFrame, template_name: str) -> pd.DataFrame:
    inputs = df["err_sentence"].tolist()
    cor_sentences = await tqdm_asyncio.gather(
        *(correct_row(template_name, text) for text in inputs),
        desc=f"🔧 [{template_name}] 문장 교정 중",
        total=len(inputs)
    )
    df["cor_sentence"] = cor_sentences
    return df

# 실행 로직
if __name__ == "__main__":
    template_name = "CHECK_SINGLE"
    input_path = "data/test.csv"
    SAMPLE_SIZE = 10871

    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY_1")

    test_df = pd.read_csv(input_path)

    is_eval_mode = (
        "answer" in test_df.columns or
        "cor_sentence" in test_df.columns or
        "cor_sentence_gt" in test_df.columns
    )

    if "answer" in test_df.columns:
        test_df = test_df.rename(columns={"answer": "cor_sentence_gt"})
    elif "cor_sentence" in test_df.columns:
        test_df = test_df.rename(columns={"cor_sentence": "cor_sentence_gt"})

    if is_eval_mode:
        sample_df = test_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        # ✅ 1. 프롬프트 토큰 수 사전 확인
        first_input = sample_df["err_sentence"].iloc[0]
        messages = apply_template(template_name, first_input)
        full_text = "\n".join([msg["content"] for msg in messages])
        token_count = get_upstage_token_count([full_text], api_key)

        print(f"🔎 현재 프롬프트 토큰 수: {token_count} tokens")
        if token_count > 5000:
            print(f"⛔ 프롬프트가 {token_count} 토큰으로 2000 토큰을 초과하여 실행을 중단합니다.")
            os._exit(1)

        # ✅ 2. 정상 추론 진행
        corrected_df = asyncio.run(run_all(sample_df, template_name))

        pred_df = corrected_df[["id", "err_sentence", "cor_sentence"]]
        true_df = sample_df[["id", "err_sentence", "cor_sentence_gt"]].rename(columns={"cor_sentence_gt": "cor_sentence"})

        print("\n📊 LCS 기반 평가 결과 (샘플 실행)")
        _ = evaluate_correction(true_df, pred_df)

        # 오답 저장
        error_df = pred_df.copy()
        error_df["ground_truth"] = true_df["cor_sentence"]
        error_df = error_df[error_df["cor_sentence"] != error_df["ground_truth"]]

        error_path = f"errors_{template_name.lower()}.csv"
        error_df.to_csv(error_path, index=False)
        print(f"❗ 틀린 샘플 {len(error_df)}개 저장됨 → {error_path}")

    else:
        corrected_df = asyncio.run(run_all(test_df, template_name))
        output_path = f"submission_{template_name.lower()}.csv"
        corrected_df[["id", "err_sentence", "cor_sentence"]].to_csv(output_path, index=False)
        print(f"\n✅ 제출 파일 저장 완료: {output_path}")

    os._exit(0)
