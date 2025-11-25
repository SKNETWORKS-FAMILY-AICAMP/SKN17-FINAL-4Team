# llm_core.py
"""
Qwen2.5-14B-Korean 기반 LLM 모듈 (8bit 양자화 로딩)

- 일반 대화 / 추천 생성: chat_template.jinja 활용 (use_chat_template=True)
- JSON 파싱(parse_user_query): 템플릿 안 쓰고 단순 텍스트 프롬프트로만 호출 (use_chat_template=False)
"""

import json
import re
from typing import List, Tuple, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from config import HF_QWEN_MODEL_NAME


# =========================
# 1. 기본 시스템 프롬프트
# =========================

DEFAULT_SYSTEM_PROMPT = (
    "너는 인테리어·홈데코 상품 추천을 도와주는 한국어 어시스턴트다. "
    "사용자의 취향(무드, 톤, 스타일), 예산, 공간 정보를 바탕으로 "
    "상품을 추천하거나 인테리어 아이디어를 제안해준다. "
    "설명은 친절하고 구체적으로, 하지만 과장 없이 해라."
)


# =========================
# 2. 모델 로딩 (8bit)
# =========================

print(f"[LLM] ▶ Qwen2.5-14B-Korean (8bit, device_map='auto') 로딩 중...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    HF_QWEN_MODEL_NAME,
    use_fast=True,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    HF_QWEN_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()


# =========================
# 3. 출력 후처리: 끊긴 문장 잘라내기
# =========================

def clean_trailing_incomplete_sentence(text: str) -> str:
    """
    생성된 텍스트가 중간에 끊긴 경우,
    마지막 '완전한 문장'까지만 남기고 나머지를 잘라낸다.

    - 1차: . ? ! … 등 문장 끝 구두점 기준
    - 2차: 한국어 종결 어미(요/합니다/입니다/예요/에요/거예요/거에요 등)를 기준으로 자르기
    - 너무 과하게 잘린 경우(전체의 50% 미만)에는 원본을 유지
    """
    text = text.strip()
    if len(text) < 40:
        # 너무 짧은 답변은 굳이 손대지 않는다.
        return text

    length = len(text)
    best_cut = -1

    # --- 1) 구두점 기준 (., ?, !, … 등) ---
    enders = [".", "?", "!", "…", "。", "！", "？"]
    last_punc_idx = -1
    for ch in enders:
        idx = text.rfind(ch)
        if idx > last_punc_idx:
            last_punc_idx = idx

    if last_punc_idx != -1 and last_punc_idx > length * 0.3:
        best_cut = max(best_cut, last_punc_idx + 1)

    # --- 2) 한국어 종결 어미 기준 ---
    # 예: "원하시나요,", "좋아요.", "괜찮습니다" 등에서 종결 어미 부분까지만 취함
    # (, . ? ! 공백 등 비-한글 문자가 뒤따르는 위치까지 포함)
    ender_pattern = re.compile(
        r"(요|입니다|합니다|예요|에요|거예요|거에요)(?=[^가-힣]|$)"
    )

    last_match_end = -1
    for m in ender_pattern.finditer(text):
        end_pos = m.end(1)
        if end_pos > last_match_end:
            last_match_end = end_pos

    if last_match_end != -1 and last_match_end > length * 0.3:
        best_cut = max(best_cut, last_match_end)

    # 자를 위치가 없으면 그대로 반환
    if best_cut == -1:
        return text

    cleaned = text[:best_cut].strip()

    # 너무 많이 잘렸으면(50% 미만 남으면) 원본 유지
    if len(cleaned) < length * 0.5:
        return text

    return cleaned


# =========================
# 4. 입력 빌더
# =========================

def _build_inputs_with_template(messages: List[Dict[str, str]]):
    """
    Qwen chat_template.jinja 를 사용한 입력 생성
    (일반 대화 / 추천 응답용)
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return {"input_ids": input_ids}


def _build_inputs_fallback(system_prompt: str, user_text: str):
    """
    chat_template 없이 단순 텍스트 프롬프트로 입력 생성
    (parse_user_query용: 버그 회피용 안전 경로)
    """
    text = (
        f"[SYSTEM]\n{system_prompt}\n\n"
        f"[USER]\n{user_text}\n\n"
        "[ASSISTANT]\n"
    )
    enc = tokenizer(text, return_tensors="pt")
    return {"input_ids": enc["input_ids"]}


# =========================
# 5. 공통 chat 함수
# =========================

def chat(
    history: List[Tuple[str, str]],
    user_input: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_chat_template: bool = True,
) -> str:
    """
    history: [(user, assistant), ...]
    user_input: 이번 턴 사용자 입력 (혹은 RAG 컨텍스트 포함 프롬프트)

    - use_chat_template=True  → Qwen chat_template 사용 (일반 대화/추천)
    - use_chat_template=False → fallback 텍스트 포맷 사용 (파서)
    """
    if use_chat_template:
        # ChatML 형식 사용
        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": system_prompt})

        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": user_input})

        inputs = _build_inputs_with_template(messages)
    else:
        # 단순 텍스트 포맷
        inputs = _build_inputs_fallback(system_prompt, user_input)

    input_ids = inputs["input_ids"]
    attention_mask = torch.ones_like(input_ids)

    # 모델의 메인 디바이스로 이동
    main_device = next(model.parameters()).device
    input_ids = input_ids.to(main_device)
    attention_mask = attention_mask.to(main_device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
        )
    else:
        gen_kwargs.update(
            do_sample=False,
        )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    input_len = input_ids.shape[1]
    generated_ids = outputs[0][input_len:]

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = text.strip()

    # 🔹 말이 애매하게 끊긴 경우, 마지막 완전한 문장까지만 남기기
    text = clean_trailing_incomplete_sentence(text)

    return text


# =========================
# 6. 사용자 질의 파싱 (카테고리/무드/예산)
# =========================

def parse_user_query(user_text: str) -> Dict[str, Any]:
    """
    Qwen에게 한 번 물어서:
    - category: "러그", "커튼", "조명", "수납장" 등 (없으면 null)
    - price_min / price_max: 원 단위 정수 (없으면 null)
    - moods: ["아늑한", "우드톤", "모던", ...] 리스트
    - space: 사용자가 꾸미고 싶다고 말한 주요 공간 (예: "책상 근처", "침실", "거실" 등)

    ⚠️ 여기서는 chat_template 사용 안 함 (HF 쪽 버그 회피용)
    """
    parse_system_prompt = (
        "너는 인테리어 상품 추천 시스템의 파서(parser)이다. "
        "사용자의 한국어 문장을 읽고 다음 정보를 JSON 형식으로만 추출해라.\n\n"
        '필드 설명:\n'
        '  - "category": 사용자가 원하는 주요 카테고리 (예: "러그", "커튼", "조명", "수납장"). 없으면 null.\n'
        '  - "price_min": 예산의 최소값 (원 단위 정수). 없으면 null.\n'
        '  - "price_max": 예산의 최대값 (원 단위 정수). 없으면 null.\n'
        '  - "moods": 사용자가 원하는 무드/분위기를 나타내는 한국어 단어 리스트 '
        '             (예: ["아늑한", "따뜻한", "미니멀", "우드톤"]).\n'
        '  - "space": 사용자가 꾸미고 싶다고 말한 주요 공간. '
        '             예: "책상 근처", "침실", "거실", "작업실", "서재", "침대 옆" 등. 없으면 null.\n\n'
        "중요 규칙:\n"
        "1) 무드(moods)에는 '아늑한', '따뜻한', '미니멀', '모던', '북유럽풍'처럼 분위기/스타일을 나타내는 형용사/형용사구만 넣어라.\n"
        "2) '책상 근처', '침실', '거실', '작업실', '서재', '방 한 구석' 같이 '공간/위치'를 나타내는 표현은 "
        "moods에 넣지 말고 반드시 space 필드에 넣어라.\n"
        "3) 사용자가 여러 공간을 말하더라도 가장 중심이 되는 한 곳만 space에 넣어라.\n"
        "4) 가격은 '10만원', '5~7만원', '20만 원 이하' 같은 표현을 적절히 해석해라. "
        "예산이 전혀 언급되지 않으면 price_min, price_max는 모두 null로 둔다.\n"
        "5) JSON 이외의 글자는 절대 출력하지 마라."
    )

    parse_user_prompt = (
        f"사용자 입력: {user_text}\n\n"
        "위 설명대로 JSON만 출력해."
    )

    raw = chat(
        history=[],
        user_input=parse_user_prompt,
        system_prompt=parse_system_prompt,
        max_new_tokens=256,
        temperature=0.7,   # do_sample=False라 실제로는 사용 안 됨
        top_p=1.0,
        do_sample=False,   # 파서는 결정적으로
        use_chat_template=False,  # 🔴 템플릿 사용 금지 (버그 회피)
    )

    # JSON 부분만 추출
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {
            "category": None,
            "price_min": None,
            "price_max": None,
            "moods": [],
            "space": None,
        }

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except Exception:
        return {
            "category": None,
            "price_min": None,
            "price_max": None,
            "moods": [],
            "space": None,
        }

    category = data.get("category")
    price_min = data.get("price_min")
    price_max = data.get("price_max")
    moods = data.get("moods") or []
    space = data.get("space")

    # moods 정제
    if isinstance(moods, str):
        moods = [m.strip() for m in moods.split(",") if m.strip()]
    elif isinstance(moods, list):
        moods = [str(m).strip() for m in moods if str(m).strip()]
    else:
        moods = []

    def _to_int_or_none(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).replace(",", "").replace(" ", "")
        if s.isdigit():
            return int(s)
        return None

    price_min = _to_int_or_none(price_min)
    price_max = _to_int_or_none(price_max)

    return {
        "category": category or None,
        "price_min": price_min,
        "price_max": price_max,
        "moods": moods,
        "space": space or None,
    }

