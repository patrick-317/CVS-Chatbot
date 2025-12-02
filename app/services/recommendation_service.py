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

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.1")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# 태그 상수
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
    """OpenAI 클라이언트 (API 키 없으면 None)."""
    global _openai_client

    if _openai_client is not None:
        return _openai_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ---------------------------------------------------------
# 타입 정의
# ---------------------------------------------------------

@dataclass
class UserPreferences:
    banned_categories: Set[str]
    diet_mode: bool
    preferred_category: Optional[str] = None
    allow_alcohol: bool = False


@dataclass
class Intent:
    mood_tags: Set[str]
    taste_tags: Set[str]
    need_alcohol: bool
    diet_mode: bool
    need_meal: bool


# ---------------------------------------------------------
# CU 상품 로딩
# ---------------------------------------------------------

def _load_cu_products() -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """CU 상품 CSV 로드 + 이름 → 상품정보 매핑."""
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
# 텍스트/이름 정규화 & 필터 유틸
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
    if p is None:
        return None
    try:
        return int(str(p).replace(",", ""))
    except Exception:
        return None


def _normalize_name_for_match(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\([^)]*\)", "", s)       # 괄호 내용 제거
    s = re.sub(r"[\s·]+", "", s)          # 공백/중점 제거
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s.lower()


def _is_food_item(name: str, main_category: Optional[str]) -> bool:
    if main_category == "생활용품":
        return False
    for kw in _NON_FOOD_KEYWORDS:
        if kw in name:
            return False
    return True


def _is_diet_friendly_items_strict(items: List[ComboItem]) -> bool:
    """다이어트 모드일 때 사용할 엄격한 필터."""
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
        # 탄수/지방도 아니고 단백질/중성도 아니면 탈락
        return False
    return has_protein


# ---------------------------------------------------------
# 콤보 CSV → 상품 매핑 + 태그/공동출현
# ---------------------------------------------------------

def _split_products_field(text: Any) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.replace(" 및 ", ",").replace(" 와 ", ",").replace("랑", ",")
    parts = re.split(r"[,/·+]|그리고|&", text)
    return [p.strip() for p in parts if p.strip()]


def _match_to_cu_product(raw_name: str, cu_norm_list: List[Tuple[str, str]]) -> Optional[str]:
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
    text = f"{keywords} {category}"
    tags: Set[str] = set()

    # 기분/상황
    if any(kw in text for kw in ["스트레스", "짜증", "열받", "화나", "빡치", "멘붕", "꿀꿀", "우울"]):
        tags.add(TAG_STRESS)
    if any(kw in text for kw in ["비 오는 날", "비오는 날", "비 오는", "빗소리", "꿀꿀한 날씨", "우중충"]):
        tags.update({TAG_RAINY, TAG_HOT_SOUP, TAG_COMFORT})

    # 맛/식사
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

    # 카테고리 기반 보정
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
    """combination + synthetic CSV를 한 번만 파싱해서 메모리에 올린다."""
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
        combo_name = str(row.get("조합 이름", row.get("combo_name", f"combo_{idx}")))
        main_text = row.get("주요 상품", row.get("main_products", ""))
        side_text = row.get("보조 상품(들)", row.get("sub_products", ""))
        keywords = str(row.get("키워드 / 상황", row.get("keywords", "")))
        category_raw = str(row.get("카테고리", row.get("category", "")))

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

        # 최소 2개 CU 상품이 매칭되는 조합만 사용
        if len(cu_items) < 2:
            continue

        combo_tags = _extract_tags_from_combo_text(keywords, category_raw)

        # 상품별 태그
        for pname in cu_items:
            product_tags.setdefault(pname, set()).update(combo_tags)

        # 공동 출현 카운트
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
# Embedding 기반 RAG-lite
# ---------------------------------------------------------

def _build_combo_embedding_text(row: Dict[str, Any]) -> str:
    parts = [
        str(row.get("name", "")),
        str(row.get("category_raw", "")),
        str(row.get("keywords", "")),
        " / ".join(row.get("product_names", [])),
    ]
    return " | ".join(p for p in parts if p)


def _ensure_combo_embeddings_built():
    """콤보 전체 임베딩을 한 번만 계산."""
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
            _combo_embeddings = []
            return

    _combo_embeddings = embeddings


def _cosine_sim(a: List[float], b: List[float]) -> float:
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
# 사용자 의도 분석 (규칙 + LLM)
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
    """단순 규칙 기반 intent 추론."""
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
        taste_tags.update({TAG_SPICY, TAG_SWEET})

    rainy_keywords = [
        "비도 오고", "비 와", "비와", "비 오", "비오는", "비 오는",
        "우중충", "우중충하네", "우울한 날",
    ]
    if any(kw in t for kw in rainy_keywords):
        mood_tags.add(TAG_RAINY)
        taste_tags.add(TAG_HOT_SOUP)

    if any(kw in t for kw in ["매운", "매콤", "얼얼"]):
        taste_tags.add(TAG_SPICY)
    if any(kw in t for kw in ["달달", "달콤", "당충전"]):
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
    """LLM을 사용한 보조 intent 추론."""
    client = _get_openai_client()
    if client is None:
        return None

    system_msg = (
        "너는 편의점 음식 추천을 위한 텍스트 분석기야. "
        "사용자의 한국어 문장을 보고 다음 항목을 JSON으로 출력해.\n"
        "- mood_tags: ['STRESS','RAINY'] 배열\n"
        "- taste_tags: ['SPICY','SWEET','HOT_SOUP','MEAL','PROTEIN','COMFORT'] 중 일부\n"
        "- need_alcohol: true/false\n"
        "- diet_mode: true/false\n"
        "- need_meal: true/false\n"
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

        return Intent(
            mood_tags=set(data.get("mood_tags", [])),
            taste_tags=set(data.get("taste_tags", [])),
            need_alcohol=bool(data.get("need_alcohol", False)),
            diet_mode=bool(data.get("diet_mode", False)),
            need_meal=bool(data.get("need_meal", False)),
        )
    except Exception:
        return None


def analyze_user_intent(text: str) -> Intent:
    """규칙 + LLM을 합쳐서 최종 intent 생성."""
    base = analyze_user_intent_rule(text)
    llm = analyze_user_intent_llm(text)

    if llm is None:
        return base

    return Intent(
        mood_tags=base.mood_tags | llm.mood_tags,
        taste_tags=base.taste_tags | llm.taste_tags,
        need_alcohol=base.need_alcohol or llm.need_alcohol,
        diet_mode=base.diet_mode or llm.diet_mode,
        need_meal=base.need_meal or llm.need_meal,
    )


def infer_category_from_text(user_text: str) -> Optional[str]:
    """문장 안에서 원하는 큰 카테고리 유추."""
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
    """라면 제외/다이어트/선호 카테고리/술 허용 여부 파싱."""
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
# 콤보 필터링 (다이어트/라면 제외/술 허용)
# ---------------------------------------------------------

def apply_negative_preferences_and_diet(
        combo: HoneyCombo,
        prefs: UserPreferences,
) -> Optional[HoneyCombo]:
    """사용자 제약을 적용해 콤보를 필터."""
    # 술안주인데 술 허용 X
    if combo.category == "술안주/야식" and not prefs.allow_alcohol:
        return None

    # 카테고리 금지
    if combo.category in prefs.banned_categories:
        return None

    # 라면/분식 금지인데 상품명에 라면/면류 포함
    if "라면/분식" in prefs.banned_categories:
        for it in combo.items:
            name = it.name or ""
            if any(re.search(pat, name) for pat in _RAMEN_NAME_PATTERNS):
                return None

    # 비식품 제거
    filtered_items: List[ComboItem] = []
    for it in combo.items:
        if _is_food_item(it.name, it.main_category):
            filtered_items.append(it)

    if len(filtered_items) < 2:
        return None

    combo.items = filtered_items
    combo.total_price = _normalize_price(sum(i.price or 0 for i in filtered_items))

    # 다이어트 모드
    if prefs.diet_mode and not _is_diet_friendly_items_strict(combo.items):
        return None

    return combo


# ---------------------------------------------------------
# 콤보 카테고리 정규화
# ---------------------------------------------------------

def _normalize_combo_category(category_raw: str) -> str:
    if "술안주" in category_raw:
        return "술안주/야식"
    if "라면" in category_raw or "분식" in category_raw:
        return "라면/분식"
    if "간편 식사" in category_raw or "식사" in category_raw:
        return "식사류"
    if "디저트" in category_raw:
        return "디저트"
    return "기타"


# ---------------------------------------------------------
# 콤보 이름 직접 조회 (이름/임베딩)
# ---------------------------------------------------------

def _find_combo_by_name_or_embedding(user_text: str) -> Optional[Dict[str, Any]]:
    """
    '밴쯔 정식', '인싸력 폭발'처럼 실제 조합 이름을 입력했을 때
    해당 콤보 row를 찾아준다.
    1순위: 이름 문자열 매칭
    2순위: 임베딩 기반 의미 유사도
    """
    _ensure_combo_knowledge_built()
    if not _combo_rows:
        return None

    t = user_text.strip()
    if not t:
        return None

    # 1) 문자열 매칭
    best_row: Optional[Dict[str, Any]] = None
    best_len = 0
    for row in _combo_rows:
        name = str(row.get("name", ""))
        if not name:
            continue

        if t == name:
            return row

        if t in name or name in t:
            if len(name) > best_len:
                best_len = len(name)
                best_row = row

    if best_row is not None:
        return best_row

    # 2) 임베딩 기반
    _ensure_combo_embeddings_built()
    if not _combo_embeddings:
        return None

    user_emb = _embed_text(t)
    if user_emb is None:
        return None

    best_sim = 0.0
    best_row = None
    for idx, row in enumerate(_combo_rows):
        if idx >= len(_combo_embeddings):
            break
        sim = _cosine_sim(user_emb, _combo_embeddings[idx])
        if sim > best_sim:
            best_sim = sim
            best_row = row

    if best_row is not None and best_sim >= 0.85:
        return best_row

    return None


# ---------------------------------------------------------
# 콤보 스코어링 + 객체 변환
# ---------------------------------------------------------

def _score_combo_for_intent(
        combo_row: Dict[str, Any],
        intent: Intent,
        prefs: UserPreferences,
        user_text: str,
        user_emb: Optional[List[float]],
        combo_emb: Optional[List[float]],
) -> float:
    """태그/카테고리/임베딩을 종합해 콤보 점수 계산."""
    tags = combo_row["tags"]
    category_raw = combo_row["category_raw"]
    keywords = combo_row["keywords"]

    score = 0.0

    # 태그 매칭
    score += len(tags & intent.taste_tags) * 2.0
    score += len(tags & intent.mood_tags) * 1.5

    # 술
    if intent.need_alcohol:
        if TAG_ALCOHOL in tags or "술안주" in category_raw:
            score += 2.0
        else:
            score -= 1.0
    else:
        if "술안주" in category_raw:
            score -= 0.5

    # 식사
    if intent.need_meal:
        if "식사" in category_raw or "라면/분식" in category_raw or TAG_MEAL in tags:
            score += 1.0

    # 선호 카테고리
    if prefs.preferred_category and prefs.preferred_category in category_raw:
        score += 1.0

    # 라면 금지
    if "라면/분식" in prefs.banned_categories and "라면" in category_raw:
        score -= 100.0

    # 텍스트 키워드 매칭
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

    # 임베딩 유사도
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
    """
    CSV row → HoneyCombo 변환.
    combo.name 은 CSV의 조합 이름을 그대로 사용한다.
    """
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

    combo = HoneyCombo(
        id=int(combo_row["id"]),
        name=combo_row["name"],
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
    """콤보 데이터셋 전체에서 현재 상황에 맞는 조합 top_k개 선택."""
    _ensure_combo_knowledge_built()
    if not _combo_rows:
        return []

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

    # 아무것도 안 맞으면 가성비 fallback
    if not scored:
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
    """LLM이 새 조합을 만들 때 사용할 후보 상품 목록."""
    _ensure_combo_knowledge_built()
    df_cu, _ = _load_cu_products()
    candidates: List[Dict[str, Any]] = []

    for row in df_cu.itertuples(index=False):
        name = str(row.name)
        main_cat = row.main_category
        price = int(row.price)

        # 술 허용 안 하면 주류 제거
        if not prefs.allow_alcohol and any(x in name for x in ["맥주", "소주", "와인", "막걸리", "하이볼"]):
            continue

        # 라면 제외
        if "라면/분식" in prefs.banned_categories:
            if any(re.search(p, name) for p in _RAMEN_NAME_PATTERNS):
                continue

        if not _is_food_item(name, main_cat):
            continue

        candidates.append({
            "name": name,
            "main_category": main_cat,
            "price": price,
            "tags": list(_product_tags.get(name, [])),
        })

    return candidates[:max_candidates]


def _build_llm_combo_prompt(
        user_text: str,
        intent: Intent,
        prefs: UserPreferences,
        products: List[Dict[str, Any]],
        max_new: int,
) -> str:
    """신규 꿀조합 생성을 위한 LLM 프롬프트."""
    rules = (
        "규칙:\n"
        "- 각 조합은 반드시 2~4개의 상품으로 구성.\n"
        "- 반드시 아래 제공된 상품 리스트에서만 선택.\n"
        "- 다이어트면 단백질/샐러드/제로 음료 중심으로.\n"
        "- 라면 제외면 면류 금지.\n"
        "- 술 언급 없으면 주류 금지.\n"
        "- 조합 이름은 짧은 한국어로.\n"
    )

    example_format = '[{"name": "조합 이름", "products": ["상품1", "상품2"]}]'

    return (
            f"사용자 입력: {user_text}\n\n"
            + rules
            + f"선호 태그: mood={list(intent.mood_tags)}, "
              f"taste={list(intent.taste_tags)}, "
              f"diet={intent.diet_mode}, meal={intent.need_meal}, "
              f"allow_alcohol={prefs.allow_alcohol}\n\n"
            + "상품 목록(JSON):\n"
            + json.dumps(products, ensure_ascii=False, indent=2)
            + f"\n\n위 상품들을 조합해 최대 {max_new}개의 꿀조합을 만들어줘.\n"
            + "반드시 아래 형식의 JSON만 출력해.\n"
            + example_format
    )


def _category_from_items(
        items: List[ComboItem],
        prefs: UserPreferences,
        intent: Intent,
) -> str:
    """신규 조합의 카테고리를 추정."""
    names = " ".join(it.name or "" for it in items)
    cats = {it.main_category or "" for it in items}

    if any(x in names for x in ["맥주", "소주", "와인", "막걸리"]):
        return "술안주/야식"
    if any("라면" in c or "분식" in c for c in cats):
        return "라면/분식"
    if any("식사" in c for c in cats):
        return "식사류"
    if any("디저트" in c or "아이스크림" in c for c in cats):
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
    GPT 기반 실시간 신규 조합 생성.
    기존 product2vec 자리를 대체하지만 시그니처는 그대로 유지.
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
                {"role": "system", "content": "CU 편의점 꿀조합을 설계하는 AI 어시스턴트야."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )
        content = resp.choices[0].message.content.strip()
        combos_json = json.loads(content)
    except Exception:
        return []

    df_cu, _ = _load_cu_products()
    cu_map = {
        str(r.name): {
            "name": r.name,
            "price": int(r.price),
            "main_category": r.main_category,
        }
        for r in df_cu.itertuples(index=False)
    }

    name_to_prod: Dict[str, Dict[str, Any]] = {p["name"]: p for p in candidate_products}

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
            prod_info = name_to_prod.get(pn) or cu_map.get(pn)
            if prod_info is None:
                continue

            name = prod_info["name"]
            main_cat = prod_info.get("main_category", "기타")
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

        category = _category_from_items(items, filters, intent)

        combo = HoneyCombo(
            id=-1000 - idx,
            name=cname,
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
# 최종 추천 로직 (컨트롤러에서 호출)
# ---------------------------------------------------------

def recommend_combos_openai_rag(
        user_text: str,
        top_k: int,
        filters: UserPreferences,
) -> List[HoneyCombo]:
    """
    1) 사용자가 실제 꿀조합 이름을 말한 경우 → 그 조합을 메인으로 반환
    2) 아니면 일반 문장 기반 추천 (의도 + Embedding + 태그 스코어)
    """
    intent = analyze_user_intent(user_text)

    # 1) 조합 이름 직접 입력 ("밴쯔 정식", "앙버터 토스트" 등)
    name_hit = _find_combo_by_name_or_embedding(user_text)
    if name_hit:
        main_combo = _build_honey_combo_from_combo_row(name_hit, user_text, intent, filters)
        if main_combo:
            others_raw = recommend_combos_from_dataset(user_text, intent, filters, top_k=top_k + 3)
            others = [c for c in others_raw if c.id != main_combo.id][:max(0, top_k - 1)]
            return [main_combo] + others

    # 2) 일반 문장 입력 → 조합 추천
    return recommend_combos_from_dataset(user_text, intent, filters, top_k=top_k)
