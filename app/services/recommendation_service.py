import os
import re
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set

import pandas as pd
from openai import OpenAI

from app.schemas.recommendation_model import HoneyCombo, ComboItem


# ---------------------------------------------------------
# 경로 / 상수 / OpenAI 설정
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CU_PRODUCTS_PATH = os.path.join(DATA_DIR, "cu_official_products.csv")
COMB_PATH = os.path.join(DATA_DIR, "combination.csv")
SYN_PATH = os.path.join(DATA_DIR, "synthetic_honey_combos_1000.csv")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# 태그
TAG_SPICY = "SPICY"
TAG_SWEET = "SWEET"
TAG_HOT_SOUP = "HOT_SOUP"
TAG_COMFORT = "COMFORT"
TAG_ALCOHOL = "ALCOHOL"
TAG_DESSERT = "DESSERT"
TAG_SNACK = "SNACK"
TAG_MEAL = "MEAL"
TAG_PROTEIN = "PROTEIN"
TAG_STRESS = "STRESS"
TAG_RAINY = "RAINY"


# ---------------------------------------------------------
# 전역 캐시 / OpenAI 클라이언트
# ---------------------------------------------------------

_cu_df: Optional[pd.DataFrame] = None
_cu_name_map: Optional[Dict[str, Dict[str, Any]]] = None

_combo_rows: Optional[List[Dict[str, Any]]] = None
_product_tags: Dict[str, Set[str]] = {}
_product_coocc: Dict[str, Dict[str, int]] = {}

_combo_embeddings: Optional[List[List[float]]] = None

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> Optional[OpenAI]:
    """OpenAI 클라이언트 반환 (API키 없으면 None)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ---------------------------------------------------------
# 기본 타입 정의
# ---------------------------------------------------------

@dataclass
class UserPreferences:
    banned_categories: Set[str]
    diet_mode: bool
    preferred_category: Optional[str] = None
    allow_alcohol: bool = False


@dataclass
class Intent:
    """사용자 발화에서 추출한 의도/상황 정보."""
    mood_tags: Set[str]
    taste_tags: Set[str]
    need_alcohol: bool
    diet_mode: bool
    need_meal: bool


# ---------------------------------------------------------
# CU 상품 로딩
# ---------------------------------------------------------

def _load_cu_products() -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """CU 상품 CSV 로드 + 이름→상품정보 매핑 생성."""
    global _cu_df, _cu_name_map
    if _cu_df is not None and _cu_name_map is not None:
        return _cu_df, _cu_name_map

    df = pd.read_csv(CU_PRODUCTS_PATH)
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    name_map: Dict[str, Dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        name_map[str(row.name)] = {
            "brand": row.brand,
            "main_category": row.main_category,
            "name": row.name,
            "price": int(row.price),
        }

    _cu_df = df
    _cu_name_map = name_map
    return _cu_df, _cu_name_map


# ---------------------------------------------------------
# 텍스트/이름 정규화 & 필터 관련 유틸
# ---------------------------------------------------------

_NON_FOOD_KEYWORDS = [
    "수면안대", "수면 안대", "인형", "키링", "피규어", "텀블러", "파일",
    "노트", "샤프", "샤프펜", "스티커", "문구", "양말",
]

_CARB_RICH_NAME_KEYWORDS = [
    "밥", "김밥", "주먹밥", "도시락", "라이스", "볶음밥", "비빔밥",
    "빵", "버거", "토스트", "파스타", "스파게티", "면", "누들",
    "떡볶이", "떡", "만두", "피자",
]

_PROTEIN_NAME_KEYWORDS = [
    "닭가슴살", "닭가슴", "닭 안심", "계란", "란", "프로틴", "단백질",
    "그릭요거트", "요거트", "두부", "콩", "참치", "연어", "고등어",
    "햄", "소시지", "소세지", "치즈",
]

_NEUTRAL_NAME_KEYWORDS = [
    "샐러드", "야채", "채소", "야채스틱", "샐러드볼", "샐러드 볼",
    "오이", "토마토", "당근", "물", "생수", "워터", "제로",
    "0kcal", "블랙커피", "아메리카노",
]

_RAMEN_NAME_PATTERNS = [
    r"라면", r"컵라면", r"볶음면", r"우동", r"라멘", r"모밀", r"쫄면", r"누들",
]

_DIET_KEYWORDS = [
    "다이어트", "살 빼", "칼로리", "살찔", "살 찔", "저칼로리", "헬스", "운동 후",
]

_ALCOHOL_TEXT_KEYWORDS = [
    "술", "맥주", "소주", "와인", "하이볼", "칵테일", "한잔", "한 잔", "막걸리",
]

_HUNGER_KEYWORDS = [
    "배고파", "배 고파", "출출", "밥 먹고 싶", "밥 뭐 먹", "공복", "식사", "한끼", "한 끼",
]


def _normalize_price(p: Optional[Any]) -> Optional[int]:
    """가격 문자열 → int 변환."""
    if p is None:
        return None
    try:
        return int(str(p).replace(",", ""))
    except Exception:
        return None


def _normalize_name_for_match(s: str) -> str:
    """상품명 매칭용 정규화(괄호/공백/기호 제거 + 소문자)."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[\s·]+", "", s)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s.lower()


def _is_food_item(name: str, main_category: Optional[str]) -> bool:
    """생활용품/문구 등을 걸러내는 비식품 판별."""
    if main_category == "생활용품":
        return False
    for kw in _NON_FOOD_KEYWORDS:
        if kw in name:
            return False
    return True


def _is_diet_friendly_items_strict(items: List[ComboItem]) -> bool:
    """다이어트 모드에서 허용할 수 있는 조합인지 검사."""
    has_protein = False
    for it in items:
        name = it.name or ""
        if any(kw in name for kw in _CARB_RICH_NAME_KEYWORDS):
            return False
        is_protein = any(kw in name for kw in _PROTEIN_NAME_KEYWORDS)
        is_neutral = any(kw in name for kw in _NEUTRAL_NAME_KEYWORDS)
        if is_protein:
            has_protein = True
            continue
        if is_neutral:
            continue
        return False
    return has_protein


# ---------------------------------------------------------
# 콤보 CSV → 상품 매핑 + 태그/공동출현 빌드
# ---------------------------------------------------------

def _split_products_field(text: Any) -> List[str]:
    """콤보 CSV의 '주요 상품/보조 상품' 필드를 개별 상품명 리스트로 분리."""
    if not isinstance(text, str):
        return []
    text = text.replace(" 및 ", ",").replace(" 와 ", ",").replace("랑", ",")
    parts = re.split(r"[,/·+]|그리고|&", text)
    return [p.strip() for p in parts if p.strip()]


def _match_to_cu_product(raw_name: str, cu_norm_list: List[Tuple[str, str]]) -> Optional[str]:
    """콤보 상품명 → CU 실제 상품명으로 매칭."""
    norm = _normalize_name_for_match(raw_name)
    if not norm:
        return None

    best_name = None
    best_score = 0
    for name, n in cu_norm_list:
        if not n:
            continue
        if norm in n or n in norm:
            score = min(len(norm), len(n))
            if score > best_score:
                best_score = score
                best_name = name

    return best_name


def _extract_tags_from_combo_text(keywords: str, category: str) -> Set[str]:
    """콤보 설명/카테고리 텍스트에서 분위기·맛 태그 추출."""
    text = f"{keywords} {category}"
    tags: Set[str] = set()

    if any(kw in text for kw in ["스트레스", "짜증", "열받", "화나", "빡치", "멘붕", "꿀꿀", "우울"]):
        tags.add(TAG_STRESS)
    if any(kw in text for kw in ["비 오는 날", "비오는 날", "비 오는", "빗소리", "꿀꿀한 날씨", "우중충"]):
        tags.update({TAG_RAINY, TAG_HOT_SOUP, TAG_COMFORT})

    if any(kw in text for kw in ["맵", "매운", "매콤", "불닭", "청양", "화끈", "마라"]):
        tags.add(TAG_SPICY)
    if any(kw in text for kw in ["달콤", "달달", "당 충전", "초코", "디저트"]):
        tags.update({TAG_SWEET, TAG_DESSERT})
    if any(kw in text for kw in ["국물", "탕", "찌개", "해장", "따끈한", "따뜻함", "추운 날"]):
        tags.update({TAG_HOT_SOUP, TAG_COMFORT})
    if any(kw in text for kw in ["술", "맥주", "소주", "막걸리", "안주", "혼술"]):
        tags.add(TAG_ALCOHOL)
    if any(kw in text for kw in ["한 끼", "한끼", "식사", "든든한", "밥", "간편 식사", "라면/분식"]):
        tags.add(TAG_MEAL)
    if any(kw in text for kw in ["단백질", "닭가슴살", "다이어트", "헬스", "운동 후"]):
        tags.add(TAG_PROTEIN)

    if "술안주" in category:
        tags.add(TAG_ALCOHOL)
    if "라면/분식" in category:
        tags.update({TAG_MEAL, TAG_SPICY})
    if "간편 식사" in category or "식사" in category:
        tags.add(TAG_MEAL)
    if "디저트" in category:
        tags.update({TAG_SWEET, TAG_DESSERT})

    return tags


def _ensure_combo_knowledge_built():
    """combination + synthetic CSV를 읽어 콤보/상품 태그/공동출현 정보 캐싱."""
    global _combo_rows, _product_tags, _product_coocc

    if _combo_rows is not None:
        return

    df_cu, _ = _load_cu_products()
    cu_norm_list = [
        (row.name, _normalize_name_for_match(row.name))
        for row in df_cu.itertuples(index=False)
    ]

    df_list = []
    if os.path.exists(COMB_PATH):
        df_list.append(pd.read_csv(COMB_PATH))
    if os.path.exists(SYN_PATH):
        df_list.append(pd.read_csv(SYN_PATH))
    if not df_list:
        _combo_rows = []
        _product_tags = {}
        _product_coocc = {}
        return

    df_combo = pd.concat(df_list, ignore_index=True)

    combo_rows: List[Dict[str, Any]] = []
    product_tags: Dict[str, Set[str]] = {}
    product_coocc: Dict[str, Dict[str, int]] = {}

    for idx, row in df_combo.iterrows():
        combo_name = str(row.get("조합 이름", f"combo_{idx}"))
        main_text = row.get("주요 상품", "")
        side_text = row.get("보조 상품(들)", "")
        keywords = str(row.get("키워드 / 상황", ""))
        category_raw = str(row.get("카테고리", ""))

        main_items = _split_products_field(main_text)
        side_items = _split_products_field(side_text)
        all_raw_items = main_items + side_items
        if len(all_raw_items) < 1:
            continue

        cu_items: List[str] = []
        for raw in all_raw_items:
            matched = _match_to_cu_product(raw, cu_norm_list)
            if matched and matched not in cu_items:
                cu_items.append(matched)

        # 실제 CU 상품 최소 2개 이상 매핑되는 조합만 사용
        if len(cu_items) < 2:
            continue

        combo_tags = _extract_tags_from_combo_text(keywords, category_raw)

        for pname in cu_items:
            product_tags.setdefault(pname, set()).update(combo_tags)

        for i in range(len(cu_items)):
            for j in range(i + 1, len(cu_items)):
                a, b = cu_items[i], cu_items[j]
                product_coocc.setdefault(a, {})
                product_coocc.setdefault(b, {})
                product_coocc[a][b] = product_coocc[a].get(b, 0) + 1
                product_coocc[b][a] = product_coocc[b].get(a, 0) + 1

        combo_rows.append(
            {
                "id": idx,
                "name": combo_name,
                "category_raw": category_raw,
                "keywords": keywords,
                "tags": combo_tags,
                "product_names": cu_items,
            }
        )

    _combo_rows = combo_rows
    _product_tags = product_tags
    _product_coocc = product_coocc


# ---------------------------------------------------------
# Embedding 기반 RAG-lite용 유틸
# ---------------------------------------------------------

def _build_combo_embedding_text(row: Dict[str, Any]) -> str:
    """임베딩용 콤보 설명 텍스트."""
    parts = [
        str(row.get("name", "")),
        str(row.get("category_raw", "")),
        str(row.get("keywords", "")),
        " / ".join(row.get("product_names", [])),
    ]
    return " | ".join(p for p in parts if p)


def _ensure_combo_embeddings_built():
    """모든 콤보에 대해 OpenAI 임베딩 캐싱."""
    global _combo_embeddings
    client = _get_openai_client()
    if client is None:
        return
    if _combo_embeddings is not None:
        return

    _ensure_combo_knowledge_built()
    if not _combo_rows:
        _combo_embeddings = []
        return

    texts = [_build_combo_embedding_text(r) for r in _combo_rows]

    embeddings: List[List[float]] = []
    # 간단하게 여러 번 나눠서 호출 (과도한 토큰 방지)
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=batch,
            )
            for d in resp.data:
                embeddings.append(d.embedding)
        except Exception:
            # 실패 시 RAG-lite는 비활성화, 기존 로직만 사용
            _combo_embeddings = []
            return

    _combo_embeddings = embeddings


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """코사인 유사도 계산."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed_text(text: str) -> Optional[List[float]]:
    client = _get_openai_client()
    if client is None:
        return None
    try:
        resp = client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=[text],
        )
        return resp.data[0].embedding
    except Exception:
        return None


# ---------------------------------------------------------
# 사용자 입력 해석 (Rule 기반 + LLM 하이브리드)
# ---------------------------------------------------------

_NEGATIVE_RAMEN_PATTERNS = [
    r"라면\s*빼",
    r"라면\s*제외",
    r"라면\s*말고",
    r"컵라면\s*빼",
    r"면\s*말고\s*밥",
    r"국물\s*라면\s*말고",
]


def analyze_user_intent_rule(text: str) -> Intent:
    """기존 규칙 기반 의도 분석."""
    t = text.strip()

    mood_tags: Set[str] = set()
    taste_tags: Set[str] = set()
    need_alcohol = False
    diet_mode = False
    need_meal = False

    low_mood_keywords = [
        "스트레스", "짜증", "열받", "화나", "빡치", "멘붕",
        "꿀꿀", "우울", "기분 별로", "기분이 별로",
    ]
    if any(kw in t for kw in low_mood_keywords):
        mood_tags.add(TAG_STRESS)
        taste_tags.update([TAG_SPICY, TAG_SWEET])

    rainy_keywords = [
        "비도 오고", "비 와", "비와", "비 오", "비오는", "비 오는",
        "우중충", "우중충하네", "우울한 날",
    ]
    if any(kw in t for kw in rainy_keywords):
        mood_tags.add(TAG_RAINY)
        taste_tags.add(TAG_HOT_SOUP)

    if "매운" in t or "매콤" in t or "얼얼" in t:
        taste_tags.add(TAG_SPICY)
    if "달달" in t or "달콤" in t or "당충전" in t:
        taste_tags.add(TAG_SWEET)

    if any(kw in t for kw in _ALCOHOL_TEXT_KEYWORDS):
        need_alcohol = True

    if any(kw in t for kw in _HUNGER_KEYWORDS):
        need_meal = True

    if any(kw in t for kw in _DIET_KEYWORDS):
        diet_mode = True

    return Intent(
        mood_tags=mood_tags,
        taste_tags=taste_tags,
        need_alcohol=need_alcohol,
        diet_mode=diet_mode,
        need_meal=need_meal,
    )


def analyze_user_intent_llm(text: str) -> Optional[Intent]:
    """OpenAI GPT를 사용한 의도 분석 (실패 시 None)."""
    client = _get_openai_client()
    if client is None:
        return None

    system_msg = (
        "너는 편의점 음식 추천을 위한 텍스트 분석기야. "
        "사용자의 한국어 문장을 보고 다음 항목을 JSON으로 출력해.\n"
        "- mood_tags: ['STRESS', 'RAINY'] 같은 배열 (없으면 빈 배열)\n"
        "- taste_tags: ['SPICY','SWEET','HOT_SOUP','MEAL','PROTEIN','COMFORT'] 중 선택\n"
        "- need_alcohol: true/false (술이 어울리는 상황인지)\n"
        "- diet_mode: true/false (다이어트/저칼로리 의도가 있는지)\n"
        "- need_meal: true/false (배고파서 식사를 원하는지)\n"
        "반드시 JSON만 출력해."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)

        mood_tags = set(data.get("mood_tags", []))
        taste_tags = set(data.get("taste_tags", []))
        need_alcohol = bool(data.get("need_alcohol", False))
        diet_mode = bool(data.get("diet_mode", False))
        need_meal = bool(data.get("need_meal", False))

        return Intent(
            mood_tags=mood_tags,
            taste_tags=taste_tags,
            need_alcohol=need_alcohol,
            diet_mode=diet_mode,
            need_meal=need_meal,
        )
    except Exception:
        return None


def analyze_user_intent(text: str) -> Intent:
    """LLM + 규칙 기반 하이브리드 의도 분석."""
    base = analyze_user_intent_rule(text)
    llm = analyze_user_intent_llm(text)

    if llm is None:
        return base

    mood_tags = base.mood_tags | llm.mood_tags
    taste_tags = base.taste_tags | llm.taste_tags
    need_alcohol = base.need_alcohol or llm.need_alcohol
    diet_mode = base.diet_mode or llm.diet_mode
    need_meal = base.need_meal or llm.need_meal

    return Intent(
        mood_tags=mood_tags,
        taste_tags=taste_tags,
        need_alcohol=need_alcohol,
        diet_mode=diet_mode,
        need_meal=need_meal,
    )


def infer_category_from_text(user_text: str) -> Optional[str]:
    """발화에서 선호 카테고리(식사/라면/안주/디저트)를 추론."""
    t = user_text.strip()
    if re.search(r"식사|한끼|끼니|밥|도시락|주먹밥|샌드위치|점심|저녁|아침", t):
        return "식사류"
    if re.search(r"라면|분식|떡볶이|우동|국물", t):
        return "라면/분식"
    if re.search(r"야식|안주|맥주|소주|술|혼술|퇴근 후", t):
        return "술안주/야식"
    if re.search(r"디저트|간식|달달|스위트|달콤", t):
        return "디저트"
    return None


def parse_user_preferences(user_text: str) -> UserPreferences:
    """사용자 발화에서 금지 카테고리/다이어트/선호 카테고리/술 허용 여부 추출."""
    text = user_text or ""

    banned_categories: List[str] = []
    if any(re.search(pat, text) for pat in _NEGATIVE_RAMEN_PATTERNS):
        banned_categories.append("라면/분식")

    diet_mode = any(kw in text for kw in _DIET_KEYWORDS)
    preferred_category = infer_category_from_text(text)
    allow_alcohol = any(kw in text for kw in _ALCOHOL_TEXT_KEYWORDS)

    if preferred_category is None:
        if any(kw in text for kw in _HUNGER_KEYWORDS) and not allow_alcohol:
            preferred_category = "식사류"

    return UserPreferences(
        banned_categories=set(banned_categories),
        diet_mode=diet_mode,
        preferred_category=preferred_category,
        allow_alcohol=allow_alcohol,
    )


# ---------------------------------------------------------
# 콤보 필터링 (다이어트/라면 제외/술 허용 등)
# ---------------------------------------------------------

def apply_negative_preferences_and_diet(
        combo: HoneyCombo,
        prefs: UserPreferences,
) -> Optional[HoneyCombo]:
    """사용자 제약 조건을 적용해 콤보를 필터링."""
    if combo.category == "술안주/야식" and not prefs.allow_alcohol:
        return None

    if combo.category in prefs.banned_categories:
        return None

    if "라면/분식" in prefs.banned_categories:
        for it in combo.items:
            name = it.name or ""
            if any(re.search(pat, name) for pat in _RAMEN_NAME_PATTERNS):
                return None

    filtered_items: List[ComboItem] = []
    for it in combo.items:
        if _is_food_item(it.name, it.main_category):
            filtered_items.append(it)

    if len(filtered_items) < 2:
        return None

    combo.items = filtered_items
    combo.total_price = _normalize_price(sum(i.price or 0 for i in filtered_items))

    if prefs.diet_mode and not _is_diet_friendly_items_strict(combo.items):
        return None

    return combo


# ---------------------------------------------------------
# 콤보 카테고리/타이틀 처리
# ---------------------------------------------------------

def _normalize_combo_category(category_raw: str) -> str:
    """콤보 CSV의 카테고리 문자열을 내부 공통 카테고리로 정규화."""
    if "술안주" in category_raw:
        return "술안주/야식"
    if "라면" in category_raw or "분식" in category_raw:
        return "라면/분식"
    if "간편 식사" in category_raw or "식사" in category_raw:
        return "식사류"
    if "디저트" in category_raw:
        return "디저트"
    return "기타"


def _build_combo_title(user_text: str, intent: Intent, original_name: str) -> str:
    """사용자 상황에 맞는 조합 이름을 생성(없으면 원래 이름 사용)."""
    text = user_text

    if any(kw in text for kw in ["스트레스", "꿀꿀", "우울", "기분 별로", "기분이 별로"]):
        if TAG_SPICY in intent.taste_tags and TAG_SWEET in intent.taste_tags:
            return "꿀꿀한 날 매콤달콤 스트레스 해소 세트"
        if TAG_SPICY in intent.taste_tags:
            return "꿀꿀한 날 매운 스트레스 해소 세트"
        if TAG_SWEET in intent.taste_tags:
            return "꿀꿀한 날 달달한 위로 세트"
        return "꿀꿀한 날 위로가 되는 한 상"

    if any(kw in text for kw in ["비도 오고", "비 와", "비와", "비 오", "비오는", "비 오는", "우중충"]):
        return "비 오는 날 따뜻한 한 끼"

    if any(kw in text for kw in ["다이어트", "칼로리", "헬스", "운동 후"]):
        return "다이어트 단백질 케어 세트"

    if any(kw in text for kw in ["배고파", "출출", "한끼", "한 끼", "밥 뭐", "밥 먹고"]):
        return "출출할 때 든든한 한 끼"

    return original_name


# ---------------------------------------------------------
# 콤보 데이터셋 기반 추천 (+ Embedding 점수 결합)
# ---------------------------------------------------------

def _score_combo_for_intent(
        combo_row: Dict[str, Any],
        intent: Intent,
        prefs: UserPreferences,
        user_text: str,
        user_emb: Optional[List[float]],
        combo_emb: Optional[List[float]],
) -> float:
    """콤보 한 개가 현재 의도/선호에 얼마나 잘 맞는지 점수화."""
    tags = combo_row["tags"]
    category_raw = combo_row["category_raw"]
    keywords = combo_row["keywords"]

    score = 0.0
    score += len(tags & intent.taste_tags) * 2.0
    score += len(tags & intent.mood_tags) * 1.5

    if intent.need_alcohol:
        if TAG_ALCOHOL in tags or "술안주" in category_raw:
            score += 2.0
        else:
            score -= 1.0
    else:
        if "술안주" in category_raw:
            score -= 0.5

    if intent.need_meal:
        if "식사" in category_raw or "라면/분식" in category_raw or TAG_MEAL in tags:
            score += 1.0

    if prefs.preferred_category and prefs.preferred_category in category_raw:
        score += 1.0

    if "라면/분식" in prefs.banned_categories and "라면" in category_raw:
        score -= 100.0

    text = user_text
    if any(kw in text for kw in ["비도 오고", "비 와", "비와", "비 오", "비오는", "비 오는"]):
        if "비 오는 날" in keywords or "비오는 날" in keywords:
            score += 2.0
        if any(kw in keywords for kw in ["국물", "탕", "찌개"]):
            score += 1.0

    if any(kw in text for kw in ["스트레스", "꿀꿀", "우울"]):
        if any(kw in keywords for kw in ["스트레스", "우울할 때", "꿀꿀한"]):
            score += 2.0
        if "매운맛" in keywords or "극강의 매운맛" in keywords:
            score += 1.0
        if any(kw in keywords for kw in ["당 충전", "초콜릿", "디저트"]):
            score += 1.0

    if user_emb is not None and combo_emb is not None:
        sim = _cosine_sim(user_emb, combo_emb)
        score += sim * 3.0

    return score


def _build_honey_combo_from_combo_row(
        combo_row: Dict[str, Any],
        user_text: str,
        intent: Intent,
        prefs: UserPreferences,
) -> Optional[HoneyCombo]:
    """콤보 한 행(row)을 HoneyCombo 객체로 변환 + 사용자 제약 적용."""
    _, cu_map = _load_cu_products()

    items: List[ComboItem] = []
    total_price = 0

    for pname in combo_row["product_names"]:
        prod = cu_map.get(pname)
        if prod is None:
            continue
        price = _normalize_price(prod["price"])
        items.append(
            ComboItem(
                original_name=pname,
                name=pname,
                price=price,
                main_category=prod["main_category"],
            )
        )
        if price:
            total_price += price

    if len(items) < 2:
        return None

    category = _normalize_combo_category(combo_row["category_raw"])
    title = _build_combo_title(user_text, intent, combo_row["name"])

    combo = HoneyCombo(
        id=int(combo_row["id"]),
        name=title,
        category=category,
        items=items,
        total_price=_normalize_price(total_price),
        mood=None,
        generated=False,
    )

    return apply_negative_preferences_and_diet(combo, prefs)


def recommend_combos_from_dataset(
        user_text: str,
        intent: Intent,
        prefs: UserPreferences,
        top_k: int = 10,
) -> List[HoneyCombo]:
    """콤보 데이터셋(실제+synthetic)에서 현재 상황에 맞는 조합을 top_k개 추천."""
    _ensure_combo_knowledge_built()
    if not _combo_rows:
        return []

    # Embedding 준비
    _ensure_combo_embeddings_built()
    user_emb = _embed_text(user_text) if _combo_embeddings else None

    scored: List[Tuple[float, Dict[str, Any], Optional[List[float]]]] = []
    for idx, row in enumerate(_combo_rows):
        combo_emb = None
        if _combo_embeddings and idx < len(_combo_embeddings):
            combo_emb = _combo_embeddings[idx]
        s = _score_combo_for_intent(row, intent, prefs, user_text, user_emb, combo_emb)
        if s > 0:
            scored.append((s, row, combo_emb))

    if not scored:
        # fallback: 가성비 기준
        fallback: List[Tuple[float, Dict[str, Any]]] = []
        for row in _combo_rows:
            if not prefs.allow_alcohol and "술안주" in row["category_raw"]:
                continue
            if "라면/분식" in prefs.banned_categories and "라면" in row["category_raw"]:
                continue
            combo = _build_honey_combo_from_combo_row(row, user_text, intent, prefs)
            if not combo:
                continue
            price = combo.total_price or 999999
            score = max(0.0, 1.0 - min(price, 20000) / 20000.0)
            fallback.append((score, row))
        scored = [(s, r, None) for (s, r) in fallback]

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)

    results: List[HoneyCombo] = []
    for _, row, _ in scored:
        combo = _build_honey_combo_from_combo_row(row, user_text, intent, prefs)
        if combo is None:
            continue
        results.append(combo)
        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------
# GPT 기반 실시간 신규 조합 생성
# ---------------------------------------------------------

def _sample_candidate_products_for_llm(
        intent: Intent,
        prefs: UserPreferences,
        max_candidates: int = 60,
) -> List[Dict[str, Any]]:
    """LLM에게 보여줄 후보 상품 리스트 생성."""
    df_cu, _ = _load_cu_products()

    candidates: List[Dict[str, Any]] = []

    for row in df_cu.itertuples(index=False):
        name = str(row.name)
        main_cat = row.main_category
        price = int(row.price)

        # 술 제약
        if not prefs.allow_alcohol and any(kw in name for kw in ["맥주", "소주", "와인", "막걸리", "하이볼"]):
            continue

        # 라면/분식 금지
        if "라면/분식" in prefs.banned_categories:
            if any(re.search(pat, name) for pat in _RAMEN_NAME_PATTERNS):
                continue

        # 비식품 제거
        if not _is_food_item(name, main_cat):
            continue

        tag_list = list(_product_tags.get(name, []))

        candidates.append(
            {
                "name": name,
                "main_category": main_cat,
                "price": price,
                "tags": tag_list,
            }
        )

    return candidates[:max_candidates]


def _build_llm_combo_prompt(
        user_text: str,
        intent: Intent,
        prefs: UserPreferences,
        products: List[Dict[str, Any]],
        max_new: int,
) -> str:
    """GPT에게 실시간 꿀조합 생성을 요청하는 프롬프트."""
    return (
            "너는 CU 편의점 상품 목록을 가지고 꿀조합 세트를 설계하는 역할이야.\n\n"
            f"사용자 문장: {user_text}\n\n"
            "아래 제약을 따라 조합을 만들어.\n"
            "- 각 조합은 2~4개의 상품으로 구성해.\n"
            "- 반드시 아래에 제공되는 상품 목록에서만 골라야 해.\n"
            "- 다이어트 모드이면 단백질/샐러드/제로 음료 위주로 조합해.\n"
            "- 라면/분식 금지이면 라면/면/우동/컵라면 등은 쓰지 마.\n"
            "- 술 관련 언급이 없으면 술/주류는 포함하지 마.\n"
            "- 조합 이름은 한국어로 간단히 지어.\n\n"
            f"선호 태그(참고용): mood_tags={list(intent.mood_tags)}, "
            f"taste_tags={list(intent.taste_tags)}, "
            f"diet_mode={intent.diet_mode}, need_meal={intent.need_meal}, "
            f"allow_alcohol={prefs.allow_alcohol}\n\n"
            "상품 목록 예시(JSON):\n"
            + json.dumps(products, ensure_ascii=False, indent=2)
            + "\n\n"
              f"이 중에서 최대 {max_new}개의 조합을 만들어.\n"
              "반드시 아래 형식의 JSON만 출력해.\n"
              '[\n'
              '  {\n'
              '    "name": "조합 이름",\n'
              '    "products": ["상품명1", "상품명2", "..."]\n'
              '  }\n'
              ']\n'
    )


def _category_from_items(items: List[Dict[str, Any]], prefs: UserPreferences, intent: Intent) -> str:
    """LLM 생성 조합의 카테고리를 간단히 추론."""
    main_cats = {it.get("main_category", "") for it in items}
    name_join = " ".join(it.get("name", "") for it in items)

    if any(kw in name_join for kw in ["맥주", "소주", "와인", "막걸리", "하이볼"]):
        return "술안주/야식"
    if any(cat for cat in main_cats if "라면" in cat or "분식" in cat):
        return "라면/분식"
    if any(cat for cat in main_cats if "식사" in cat):
        return "식사류"
    if any(cat for cat in main_cats if "디저트" in cat or "아이스크림" in cat):
        return "디저트"
    if prefs.preferred_category:
        return prefs.preferred_category
    if intent.need_meal:
        return "식사류"
    return "기타"


def generate_combos_product2vec(
        user_text: str,
        base_candidates: List[HoneyCombo],
        max_new: int,
        filters: UserPreferences,
) -> List[HoneyCombo]:
    """
    GPT를 사용한 실시간 신규 조합 생성.
    - 기존 product2vec 자리는 유지하면서, LLM 기반 생성으로 대체.
    - 실패 시 빈 리스트 반환 (기존 RAG 기반 추천만 사용).
    """
    client = _get_openai_client()
    if client is None or max_new <= 0:
        return []

    intent = analyze_user_intent(user_text)
    candidate_products = _sample_candidate_products_for_llm(intent, filters)
    if not candidate_products:
        return []

    prompt = _build_llm_combo_prompt(user_text, intent, filters, candidate_products, max_new)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "CU 편의점 꿀조합 추천을 설계하는 AI 어시스턴트야."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()
        combos_json = json.loads(content)
    except Exception:
        return []

    name_to_prod: Dict[str, Dict[str, Any]] = {p["name"]: p for p in candidate_products}
    _, cu_map = _load_cu_products()

    results: List[HoneyCombo] = []

    for idx, c in enumerate(combos_json):
        if len(results) >= max_new:
            break

        try:
            cname = str(c.get("name", f"generated_combo_{idx}"))
            prod_names = [str(n) for n in c.get("products", [])]
        except Exception:
            continue

        items: List[ComboItem] = []
        total_price = 0

        for pn in prod_names:
            prod_info = name_to_prod.get(pn)
            if prod_info is None:
                cu_info = cu_map.get(pn)
                if cu_info is None:
                    continue
                prod_info = cu_info

            name = prod_info["name"]
            main_cat = prod_info.get("main_category", prod_info.get("main_category", "기타"))
            price = _normalize_price(prod_info["price"])

            items.append(
                ComboItem(
                    original_name=name,
                    name=name,
                    price=price,
                    main_category=main_cat,
                )
            )
            if price:
                total_price += price

        if len(items) < 2:
            continue

        category = _category_from_items(
            [{"name": it.name, "main_category": it.main_category} for it in items],
            filters,
            intent,
        )
        title = _build_combo_title(user_text, intent, cname)

        combo = HoneyCombo(
            id=-1000 - idx,
            name=title,
            category=category,
            items=items,
            total_price=_normalize_price(total_price),
            mood=None,
            generated=True,
        )

        combo = apply_negative_preferences_and_diet(combo, filters)
        if combo is None:
            continue

        results.append(combo)

    return results


# ---------------------------------------------------------
# 컨트롤러에서 사용하는 공개 함수
# ---------------------------------------------------------

def recommend_combos_openai_rag(
        user_text: str,
        top_k: int,
        filters: UserPreferences,
) -> List[HoneyCombo]:
    """
    combo CSV 기반 추천 엔진:
    - LLM + 규칙 기반 Intent 분석
    - Embedding 기반 의미 유사도(RAG-lite) + 태그 기반 스코어 결합
    """
    intent = analyze_user_intent(user_text)
    return recommend_combos_from_dataset(user_text, intent, filters, top_k=top_k)
