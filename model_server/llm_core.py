# llm_core.py
"""
Qwen2.5-14B-Korean 기반 LLM 모듈 (8bit 양자화 로딩)

- 일반 대화 / 추천 생성: chat_template.jinja 활용 (use_chat_template=True)
- JSON 파싱(parse_user_query): 템플릿 안 쓰고 단순 텍스트 프롬프트로만 호출 (use_chat_template=False)
"""

import json
import re
import time
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils import logging as hf_logging

from config import HF_QWEN_MODEL_NAME
from mood_vocab import snap_moods_to_vocab, match_moods_in_text  # 텍스트에서 무드 탐지


# =========================
# 1. 기본 시스템 프롬프트 (추천 전용: 강한 비환각 규칙)
# =========================

DEFAULT_SYSTEM_PROMPT = (
    "너는 인테리어·홈데코 상품 추천을 도와주는 한국어 어시스턴트다.\n"
    "사용자의 취향(무드, 톤, 스타일), 예산, 공간 정보를 바탕으로 "
    "상품을 추천하거나 인테리어 아이디어를 제안한다.\n\n"
    "[역할]\n"
    "- 사용자가 꾸미고 싶은 공간의 현재 분위기와 목표 분위기를 이해하고,\n"
    "  시스템이 제공하는 '후보 상품 목록' 안에서만 상품을 골라 추천한다.\n\n"
    "[가장 중요한 규칙 – 반드시 지켜야 함]\n"
    "1) 시스템이 제공한 후보 상품 목록(브랜드, 상품명, 가격, 링크) 밖의 상품을 "
    "   절대로 새로 만들어 내지 마라.\n"
    "   - 새로운 브랜드 이름을 지어내지 마라.\n"
    "   - 새로운 상품명을 지어내지 마라.\n"
    "   - 새로운 가격이나 링크(URL)를 지어내지 마라.\n"
    "   - 후보 목록에 없는 구체적인 제품을 예시로 들지 마라.\n"
    "2) 답변에 등장하는 모든 '브랜드/상품명/가격/링크'는 반드시 입력으로 받은 텍스트에서\n"
    "   그대로 복사해 사용해야 한다. 오타를 수정하거나 의미를 바꾸지 마라.\n"
    "3) 후보 목록에 적당한 상품이 없으면, 없는 상태를 솔직하게 설명하고\n"
    "   '현재 데이터에 없는 상품입니다'처럼 말해라. 절대로 상상으로 채우지 마라.\n"
    "4) 제품이 아닌 개념적인 조언(예: \"우드톤 소품을 추가해보세요\")은 할 수 있지만,\n"
    "   그 경우에는 특정 브랜드명/상품명/가격/링크를 절대 붙이지 마라.\n"
    "5) 요약/설명 문장에서는 자연스럽게 말을 해도 되지만,\n"
    "   실제로 언급되는 상품 정보는 항상 후보 목록에서 온 것인지 스스로 다시 확인해라.\n\n"
    "[답변 스타일]\n"
    "- 톤: 인테리어 전문가가 친절하게 조언해 주는 느낌.\n"
    "- 길이: 너무 장황하지 않게, 그러나 사용자가 실제로 참고할 수 있을 정도로 충분히 구체적으로.\n"
    "- 구조: 현재 공간 설명 → 목표 무드 요약 → 후보 상품들 중 일부를 선택해\n"
    "        어떻게 배치하면 좋은지 단계적으로 설명.\n"
)


# =========================
# 2. 모델 로딩 (8bit)
# =========================

hf_logging.set_verbosity_error()

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
# 3. 출력 후처리
# =========================

def clean_trailing_incomplete_sentence(text: str) -> str:
    """
    생성된 텍스트가 중간에 끊긴 경우,
    마지막 '완전한 문장'까지만 남기고 나머지를 잘라낸다.
    """
    text = text.strip()
    if len(text) < 40:
        return text

    length = len(text)
    best_cut = -1

    # 1) 구두점 기준
    enders = [".", "?", "!", "…", "。", "！", "？"]
    last_punc_idx = -1
    for ch in enders:
        idx = text.rfind(ch)
        if idx > last_punc_idx:
            last_punc_idx = idx

    if last_punc_idx != -1 and last_punc_idx > length * 0.3:
        best_cut = max(best_cut, last_punc_idx + 1)

    # 2) 한국어 종결 어미 기준
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

    if best_cut == -1:
        return text

    cleaned = text[:best_cut].strip()
    if len(cleaned) < length * 0.5:
        return text

    return cleaned


# =========================
# 4. 입력 빌더
# =========================

def _build_inputs_with_template(messages: List[Dict[str, str]]):
    """Qwen chat_template.jinja 를 사용한 입력 생성 (일반 대화용)."""
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
    (parse_user_query용: 버그 회피용)
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
    max_new_tokens: int = 1024,
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

    ⚠️ 주의:
    - main.py에서 추천 모드(handle_recommend)는 system_prompt로 DEFAULT_SYSTEM_PROMPT를 넘긴다.
      이 경우를 '상품 추천/비환각 모드'로 간주하여 샘플링 파라미터를 조금 더 보수적으로 조정한다.
    """
    # 추천 모드 여부 판단 (DEFAULT_SYSTEM_PROMPT 그대로 쓴 경우)
    is_recommendation_mode = (system_prompt == DEFAULT_SYSTEM_PROMPT)

    # 추천 모드에서는 온도를 조금 낮추고(top_p도 살짝 낮춤) 더 보수적으로 답변
    effective_temperature = temperature
    effective_top_p = top_p
    if is_recommendation_mode:
        # 지나치게 창의적인(=환각) 출력을 줄이기 위해 상한을 둔다.
        effective_temperature = min(float(temperature), 0.5)
        effective_top_p = min(float(top_p), 0.85)

    if use_chat_template:
        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": system_prompt})

        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": user_input})
        inputs = _build_inputs_with_template(messages)
    else:
        inputs = _build_inputs_fallback(system_prompt, user_input)

    input_ids = inputs["input_ids"]
    attention_mask = torch.ones_like(input_ids)

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
            temperature=float(effective_temperature),
            top_p=float(effective_top_p),
        )
    else:
        gen_kwargs.update(do_sample=False)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    t1 = time.time()
    _elapsed = t1 - t0  # 내부에서는 로그만 제거, 값은 필요하면 디버그용으로 남겨둘 수 있음

    input_len = input_ids.shape[1]
    generated_ids = outputs[0][input_len:]

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = text.strip()
    text = clean_trailing_incomplete_sentence(text)

    # 🔇 여기서는 시간 로그 출력 안 함 (main.py에서만 출력)
    return text


# =========================
# 6. 사용자 질의 파싱 (카테고리/무드/예산/공간)
# =========================

def parse_user_query(user_text: str) -> Dict[str, Any]:
    """
    Qwen에게 한 번 물어서:
    - category: "러그", "커튼", "조명", "수납장" 등 (없으면 null)
    - price_min / price_max: 원 단위 정수 (없으면 null)
    - moods: ["아늑한", "우드톤", "모던", ...] 리스트
    - space: "침실", "거실", "책상 근처" 등

    + 여러 휴리스틱으로 보정 (예산/공간/무드/카테고리).
    """

    # ---------- 0) 휴리스틱들 ----------

    def _heuristic_detect_space(text: str) -> Optional[str]:
        if "침실" in text:
            return "침실"
        if "거실" in text:
            return "거실"
        if "작업실" in text or "공부방" in text or "서재" in text:
            return "작업실/서재"
        if "책상" in text:
            return "책상 근처"
        if "주방" in text or "부엌" in text:
            return "주방"
        return None

    def _heuristic_detect_moods(text: str) -> List[str]:
        candidates: List[str] = []

        mood_patterns = [
            ("차분", "차분한"),
            ("잔잔", "차분한"),
            ("따뜻", "따뜻한"),
            ("포근", "포근한"),
            ("아늑", "아늑한"),
            ("편안", "편안한"),
            ("모던", "모던"),
            ("현대적", "모던"),
            ("심플", "미니멀"),
            ("미니멀", "미니멀"),
            ("북유럽", "북유럽풍"),
            ("호텔", "호텔식"),
            ("우드톤", "우드톤"),
            ("화이트톤", "화이트톤"),
        ]

        for key, label in mood_patterns:
            if key in text:
                candidates.append(label)

        return list(dict.fromkeys(candidates))

    def _heuristic_detect_budget(text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        예산 관련 휴리스틱 파서.
        """
        m_range2 = re.search(
            r"(\d+)\s*만\s*원?\s*(?:이상|초과|부터)[^0-9]{0,15}(\d+)\s*만\s*원?\s*(?:이하|이내|까지|언더|아래|밑)",
            text,
        )
        if m_range2:
            a = int(m_range2.group(1)) * 10000
            b = int(m_range2.group(2)) * 10000
            return (min(a, b), max(a, b))

        m_range = re.search(
            r"(\d+)\s*만\s*원?\s*(?:에서|~|-)\s*(\d+)\s*만",
            text,
        )
        if m_range:
            a = int(m_range.group(1)) * 10000
            b = int(m_range.group(2)) * 10000
            return (min(a, b), max(a, b))

        m_around_num = re.search(
            r"(\d+)\s*만\s*원?\s*(?:정도|쯤|전후|근처|근방|언저리)",
            text,
        )
        if m_around_num:
            v = int(m_around_num.group(1)) * 10000
            lo = int(v * 0.8)
            hi = int(v * 1.2)
            return (lo, hi)

        matches = [
            (int(m.group(1)) * 10000, m.start())
            for m in re.finditer(r"(\d+)\s*만\s*원?", text)
        ]
        if not matches:
            return (None, None)

        cap_words = ["이내", "이하", "까지", "최대", "언더", "아래", "밑"]
        low_words = ["이상", "부터", "넘게", "초과", "오버", "위"]
        around_words = ["정도", "쯤", "전후", "근처", "근방", "언저리"]

        def first_pos(words: List[str]) -> int:
            poss = [text.find(w) for w in words if w in text]
            return min(poss) if poss else -1

        cap_pos = first_pos(cap_words)
        low_pos = first_pos(low_words)
        around_pos = first_pos(around_words)

        def pick_before(pos: int) -> int:
            if pos == -1:
                return matches[-1][0]
            before = [m for m in matches if m[1] <= pos]
            return before[-1][0] if before else matches[-1][0]

        if cap_pos != -1:
            v = pick_before(cap_pos)
            return (None, v)

        if low_pos != -1:
            v = pick_before(low_pos)
            return (v, None)

        v = matches[-1][0]
        if around_pos != -1:
            lo = int(v * 0.8)
            hi = int(v * 1.2)
            return (lo, hi)

        lo = int(v * 0.7)
        hi = int(v * 1.3)
        return (lo, hi)

    def _looks_like_interior_context(text: str) -> bool:
        interior_words = [
            "인테리어", "집 꾸미", "집꾸미",
            "공간", "방", "거실", "침실", "작업실", "서재",
            "가구", "소품", "쿠션", "러그", "조명", "커튼",
        ]
        return any(w in text for w in interior_words)

    def _heuristic_detect_category(text: str) -> Optional[str]:
        t = text.replace(" ", "").lower()

        if any(k in t for k in ["조명", "램프", "무드등", "스탠드", "벽걸이조명", "벽등", "백열등"]):
            return "조명"
        if any(k in t for k in ["러그", "카페트", "카펫", "카펫트"]):
            return "러그_커튼"
        if any(k in t for k in ["커튼", "블라인드"]):
            return "러그_커튼"
        if any(k in t for k in ["쿠션", "쿠션커버", "방석"]):
            return "쿠션"
        if any(k in t for k in ["이불", "침구", "베딩", "이불커버", "침대커버"]):
            return "침구"
        if any(k in t for k in ["선반", "수납", "서랍", "책장", "수납장"]):
            return "수납정리"
        return None

    def _normalize_category_str(cat: Optional[str]) -> Optional[str]:
        if not cat:
            return None
        s = str(cat).strip().lower()

        mapping = {
            "lighting": "조명",
            "light": "조명",
            "lamp": "조명",
            "rug": "러그_커튼",
            "curtain": "러그_커튼",
            "carpet": "러그_커튼",
            "bedding": "침구",
            "blanket": "침구",
            "duvet": "침구",
            "pillow": "쿠션",
            "cushion": "쿠션",
            "storage": "수납정리",
            "shelf": "수납정리",
        }

        if s in mapping:
            return mapping[s]

        if re.search(r"[가-힣]", s):
            return s
        return s

    # ---------- 1) LLM 기반 1차 파싱 ----------

    parse_system_prompt = (
        "너는 인테리어 상품 추천 시스템의 파서(parser)이다. "
        "사용자의 한국어 문장을 읽고 다음 정보를 JSON 형식으로만 추출해라.\n\n"
        '필드 설명:\n'
        '  - "category": 사용자가 원하는 주요 카테고리 (예: "러그", "커튼", "조명", "수납장"). 없으면 null.\n'
        '  - "price_min": 예산의 최소값 (원 단위 정수). 없으면 null.\n'
        '  - "price_max": 예산의 최대값 (원 단위 정수). 없으면 null.\n'
        '  - "moods": 사용자가 원하는 무드/분위기를 나타내는 한국어 단어 리스트.\n'
        '  - "space": 사용자가 꾸미고 싶다고 말한 주요 공간. 예: "책상 근처", "침실", "거실", "작업실" 등.\n\n'
        "중요 규칙:\n"
        "1) 무드(moods)에는 분위기/스타일을 나타내는 표현만 넣어라.\n"
        "2) '책상 근처', '침실', '거실' 같은 공간 표현은 space에만 넣고 moods에는 넣지 마라.\n"
        "3) 예산이 전혀 언급되지 않으면 price_min, price_max는 모두 null로 둔다.\n"
        "4) JSON 이외의 글자는 절대 출력하지 마라."
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
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        use_chat_template=False,
    )

    # ---------- 2) JSON 부분만 추출 ----------

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        data = {}
    else:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
        except Exception:
            data = {}

    # ---------- 3) 1차 추출 값 ----------

    category = data.get("category")
    price_min = data.get("price_min")
    price_max = data.get("price_max")
    moods = data.get("moods") or []
    space = data.get("space")

    # ---------- 4) 타입/포맷 정리 ----------

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

    text = user_text.strip()

    # ---------- 4-1) 무드 사전에 기반한 직접 탐지 ----------

    # 사용자의 원문 문장에서 사전에 있는 무드 키워드를 직접 찾아서
    # LLM이 뽑은 moods 리스트에 합쳐준다.
    detected_moods_from_text = match_moods_in_text(text)
    for m in detected_moods_from_text:
        if m not in moods:
            moods.append(m)

    # ---------- 5) 예산 휴리스틱: 둘 다 None일 때만 ----------

    if price_min is None and price_max is None:
        h_min, h_max = _heuristic_detect_budget(text)
        if h_min is not None or h_max is not None:
            price_min, price_max = h_min, h_max

    # ---------- 6) 인테리어 문맥일 때 space/moods 보정 ----------

    if _looks_like_interior_context(text):
        if space is None:
            h_space = _heuristic_detect_space(text)
            if h_space:
                space = h_space

        # LLM/사전 둘 다 못 잡았을 때만 휴리스틱으로 채움
        if not moods:
            h_moods = _heuristic_detect_moods(text)
            if h_moods:
                moods = h_moods

    # ---------- 7) 무드 정규화 + '존재하지 않는 무드' 검출 ----------

    canonical_moods, unknown_moods = snap_moods_to_vocab(moods)

    # 🔹 무드로 쓰면 안 되는 단어들 정리 (사진은 이미지를 가리키는 말일 뿐)
    BAD_MOOD_TOKENS = {"소품", "사진", "사진같은", "사진 같은", "이미지", "그림"}

    # 🔹 canonical 무드만 진짜 moods로 인정 + BAD_MOOD_TOKENS 제거
    moods = [m for m in canonical_moods if m not in BAD_MOOD_TOKENS]
    unknown_moods = [m for m in unknown_moods if m not in BAD_MOOD_TOKENS]

    # ---------- 8) 카테고리 보정 ----------

    category = _normalize_category_str(category)

    if category is None:
        h_cat = _heuristic_detect_category(text)
        if h_cat:
            category = h_cat

    return {
        "category": category or None,
        "price_min": price_min,
        "price_max": price_max,
        "moods": moods,
        "space": space or None,
        # 사전에 없는 무드 표현 리스트
        "unknown_moods": unknown_moods,
    }
