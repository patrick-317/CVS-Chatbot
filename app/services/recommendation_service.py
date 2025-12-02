import os
import re
import math
import json
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
_cu_products: Optional[List[Dict[str, Any]]] = None          # 전체 상품 리스트
_product_embeddings: Optional[Dict[str, List[float]]] = None  # 상품명 -> embedding
_product_tags: Dict[str, Set[str]] = {}                       # 상품명 -> 태그들
_product_coocc: Dict[str, Dict[str, int]] = {}                # 상품명 -> {다른상품: count}

_combo_name_index: Optional[List[Dict[str, Any]]] = None      # 조합 이름 검색용 (밴쯔정식 등)

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
# 기본 유틸 (정규화 / 필터)
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
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[\s·]+", "", s)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s.lower()


def _is_food_item(name: str, main_category: Optional[str]) -> bool:
    if main_category and "생활용품" in main_category:
        return False
    for kw in _NON_FOOD_KEYWORDS:
        if kw in name:
            return False
    return True


def _is_diet_friendly_items_strict(items: List[ComboItem]) -> bool:
    """단백질+중성 위주인지, 탄수/디저트가 거의 없는지 체크."""
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
# CU 상품 로딩 / 태깅 / 공동출현
# ---------------------------------------------------------

def _load_cu_products() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """CU 상품 CSV 로드 + 리스트 캐시."""
    global _cu_df, _cu_products
    if _cu_df is not None and _cu_products is not None:
        return _cu_df, _cu_products

    df = pd.read_csv(CU_PRODUCTS_PATH)
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    products: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        products.append(
            {
                "name": str(row.name),
                "brand": str(getattr(row, "brand", "")),
                "main_category": str(getattr(row, "main_category", "")),
                "price": int(row.price),
            }
        )

    _cu_df = df
    _cu_products = products
    return _cu_df, _cu_products


def _split_products_field(text: Any) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.replace(" 및 ", ",").replace(" 와 ", ",").replace("랑", ",")
    parts = re.split(r"[,/·+]|그리고|&", text)
    return [p.strip() for p in parts if p.strip()]


def _match_to_cu_product(raw_name: str, cu_norm_list: List[Tuple[str, str]]) -> Optional[str]:
    """콤보 CSV 문자열 상품명을 CU 상품명으로 매칭."""
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
    """콤보 CSV의 키워드/카테고리에서 분위기/맛 태그 추출."""
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


def _ensure_product_stats_built():
    """
    상품 태그(product_tags) / 공동출현(product_coocc) / 콤보 이름 인덱스(_combo_name_index) 구축.
    - combination.csv + synthetic_honey_combos_1000.csv 기반
    - '밴쯔 정식', '앙버터 토스트' 같은 조합 이름 인식용
    """
    global _product_tags, _product_coocc, _combo_name_index

    if _combo_name_index is not None:
        return

    _, products = _load_cu_products()
    cu_name_to_norm = {
        p["name"]: _normalize_name_for_match(p["name"]) for p in products
    }
    cu_norm_list = list(cu_name_to_norm.items())  # (name, norm)

    df_list: List[pd.DataFrame] = []
    if os.path.exists(COMB_PATH):
        df_list.append(pd.read_csv(COMB_PATH))
    if os.path.exists(SYN_PATH):
        df_list.append(pd.read_csv(SYN_PATH))

    if not df_list:
        _product_tags = {}
        _product_coocc = {}
        _combo_name_index = []
        return

    df_combo = pd.concat(df_list, ignore_index=True)

    combo_name_index: List[Dict[str, Any]] = {}
    product_tags: Dict[str, Set[str]] = {}
    product_coocc: Dict[str, Dict[str, int]] = {}

    combo_rows: List[Dict[str, Any]] = []

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
                "product_names": cu_items,
            }
        )

    # 콤보 이름 검색용 인덱스
    combo_name_index = []
    for row in combo_rows:
        combo_name_index.append(
            {
                "id": row["id"],
                "name": row["name"],
                "product_names": row["product_names"],
                "category_raw": row["category_raw"],
            }
        )

    _product_tags = product_tags
    _product_coocc = product_coocc
    _combo_name_index = combo_name_index


# ---------------------------------------------------------
# Embedding 유틸 (상품 / 사용자 문장)
# ---------------------------------------------------------

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


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _ensure_product_embeddings_built():
    """CU 상품 전체 embedding 1회 계산 후 캐시."""
    global _product_embeddings
    if _product_embeddings is not None:
        return

    client = _get_openai_client()
    if client is None:
        _product_embeddings = {}
        return

    _, products = _load_cu_products()
    texts: List[str] = []
    keys: List[str] = []

    for p in products:
        name = p["name"]
        main_cat = p["main_category"]
        brand = p["brand"]
        txt = f"{brand} {name} ({main_cat})"
        texts.append(txt)
        keys.append(name)

    embeddings: Dict[str, List[float]] = {}
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=batch,
            )
        except Exception:
            # 임베딩 실패하면 그냥 rule 기반만 사용
            _product_embeddings = {}
            return
        for j, d in enumerate(resp.data):
            embeddings[keys[i + j]] = d.embedding

    _product_embeddings = embeddings


# ---------------------------------------------------------
# 사용자 의도 분석 (키워드 기반)
# ---------------------------------------------------------

_NEGATIVE_RAMEN_PATTERNS = [
    r"라면\s*빼",
    r"라면\s*제외",
    r"라면\s*말고",
    r"컵라면\s*빼",
    r"면\s*말고\s*밥",
    r"국물\s*라면\s*말고",
]


def analyze_user_intent(text: str) -> Intent:
    """기분/상황/맛/식사 여부 등을 태그로 변환."""
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


def infer_category_from_text(user_text: str) -> Optional[str]:
    """문장에서 큰 카테고리(식사/라면/야식/디저트 등) 유추."""
    t = user_text.strip()
    if re.search(r"식사|한끼|끼니|밥|도시락|주먹밥|샌드위치|점심|저녁|아침", t):
        return "식사류"
    if re.search(r"라면|분식|떡볶이|우동|국물", t):
        return "라면/분식"
    if re.search(r"야식|안주|맥주|소주|술|혼술|퇴근 후", t):
        return "술안주/야식"
    if re.search(r"디저트|간식|달달|스위트|달콤", t):
        return "디저트"
    if re.search(r"다이어트|칼로리|저칼로리|헬스", t):
        return "다이어트/건강"
    return None


def parse_user_preferences(user_text: str) -> UserPreferences:
    """라면 제외 / 다이어트 / 선호 카테고리 / 술 허용 여부 파싱."""
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
# 콤보 이름 직접 조회 (밴쯔 정식, 앙버터 토스트 등)
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


def _find_combo_by_name(user_text: str) -> Optional[Dict[str, Any]]:
    """
    '밴쯔 정식', '앙버터 토스트'처럼
    실제 CSV에 있는 조합 이름을 입력했을 때 매칭.
    (단순 문자열 기반)
    """
    _ensure_product_stats_built()
    if not _combo_name_index:
        return None

    t = user_text.strip()
    if not t:
        return None

    best_row = None
    best_len = 0
    for row in _combo_name_index:
        name = str(row["name"])
        if not name:
            continue
        if t == name:
            return row
        if t in name or name in t:
            if len(name) > best_len:
                best_len = len(name)
                best_row = row

    return best_row


def _build_honey_combo_from_named_combo(
        combo_row: Dict[str, Any],
        prefs: UserPreferences,
) -> Optional[HoneyCombo]:
    """콤보 CSV row → HoneyCombo (이름 그대로 유지)."""
    _, products = _load_cu_products()
    cu_map = {p["name"]: p for p in products}

    items: List[ComboItem] = []
    total_price = 0

    for pname in combo_row["product_names"]:
        prod = cu_map.get(pname)
        if not prod:
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


# ---------------------------------------------------------
# 콤보 필터링 (다이어트 / 라면 제외 / 술 허용)
# ---------------------------------------------------------

def apply_negative_preferences_and_diet(
        combo: HoneyCombo,
        prefs: UserPreferences,
) -> Optional[HoneyCombo]:
    """사용자 제약(술/라면/다이어트/비식품)을 적용해 콤보 필터링."""
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
# 상품 단위 점수 계산 / 콤보 생성
# ---------------------------------------------------------

def _score_product_for_intent(
        product: Dict[str, Any],
        tags: Set[str],
        intent: Intent,
        prefs: UserPreferences,
        user_emb: Optional[List[float]],
        prod_emb: Optional[List[float]],
) -> float:
    """상품 1개에 대한 점수 계산."""
    score = 0.0
    name = product["name"]
    main_cat = product["main_category"]

    score += len(tags & intent.taste_tags) * 2.0
    score += len(tags & intent.mood_tags) * 1.5

    if intent.need_alcohol:
        if TAG_ALCOHOL in tags or "주류" in main_cat:
            score += 2.0
        else:
            score -= 1.0
    else:
        if "주류" in main_cat:
            score -= 2.0

    if intent.need_meal:
        if any(kw in main_cat for kw in ["간편식", "식사"]) or any(
                kw in name for kw in ["도시락", "김밥", "주먹밥", "덮밥", "파스타", "라면"]
        ):
            score += 1.5

    if prefs.preferred_category == "라면/분식":
        if any(kw in name for kw in ["라면", "우동", "떡볶이", "면"]):
            score += 1.5
    if prefs.preferred_category == "식사류":
        if any(kw in name for kw in ["도시락", "김밥", "주먹밥", "덮밥"]):
            score += 1.5
    if prefs.preferred_category == "디저트":
        if TAG_DESSERT in tags or TAG_SWEET in tags:
            score += 1.5

    if "라면/분식" in prefs.banned_categories:
        if any(re.search(pat, name) for pat in _RAMEN_NAME_PATTERNS):
            score -= 100.0

    if prefs.diet_mode:
        if TAG_PROTEIN in tags:
            score += 1.0
        if TAG_DESSERT in tags or TAG_SWEET in tags:
            score -= 1.0
        if any(kw in name for kw in _CARB_RICH_NAME_KEYWORDS):
            score -= 1.0

    if user_emb is not None and prod_emb is not None:
        sim = _cosine_sim(user_emb, prod_emb)
        score += sim * 2.5

    return score


def _category_from_products(
        products: List[Dict[str, Any]],
        prefs: UserPreferences,
        intent: Intent,
) -> str:
    """조합에 포함된 상품들을 보고 콤보 카테고리 추정."""
    names = " ".join(p["name"] for p in products)
    cats = " ".join(p["main_category"] for p in products)

    if any(kw in names for kw in ["맥주", "소주", "와인", "막걸리", "하이볼"]) or "주류" in cats:
        return "술안주/야식"
    if any(kw in names for kw in ["라면", "우동", "떡볶이", "면"]):
        return "라면/분식"
    if any(kw in names for kw in ["도시락", "김밥", "주먹밥", "덮밥", "파스타"]):
        return "식사류"
    if any(kw in names for kw in ["케이크", "케익", "쿠키", "젤리", "아이스크림", "빙수"]):
        return "디저트"
    if prefs.preferred_category:
        return prefs.preferred_category
    if intent.need_meal:
        return "식사류"
    return "기타"


def _build_combo_title_from_intent(
        user_text: str,
        intent: Intent,
        category: str,
) -> str:
    """의도/카테고리 기반 콤보 이름 생성 (규칙 기반)."""
    t = user_text

    if any(kw in t for kw in ["스트레스", "짜증", "열받", "꿀꿀", "우울"]):
        if TAG_SPICY in intent.taste_tags and TAG_SWEET in intent.taste_tags:
            return "매콤달콤 스트레스 해소 세트"
        if TAG_SPICY in intent.taste_tags:
            return "매운맛으로 날려버리는 스트레스 해소 세트"
        if TAG_SWEET in intent.taste_tags:
            return "달콤하게 위로해주는 스트레스 해소 세트"
        return "스트레스 해소 든든 세트"

    if any(kw in t for kw in ["비도 오고", "비 와", "비와", "비 오는", "우중충"]):
        if category == "라면/분식" or TAG_HOT_SOUP in intent.taste_tags:
            return "비 오는 날 따뜻한 국물 세트"
        return "비 오는 날 출출할 때 세트"

    if any(kw in t for kw in ["다이어트", "칼로리", "헬스", "운동 후"]):
        return "다이어트 단백질 케어 세트"

    if any(kw in t for kw in ["배고파", "출출", "한끼", "한 끼", "밥 뭐", "밥 먹고"]):
        return "출출할 때 든든한 식사 세트"

    if category == "술안주/야식":
        return "오늘 밤 혼술 안주 세트"
    if category == "디저트":
        return "달콤한 디저트 타임 세트"

    return "편의점 꿀조합 세트"


def _build_combos_from_products(
        user_text: str,
        intent: Intent,
        prefs: UserPreferences,
        top_k: int,
) -> List[HoneyCombo]:
    """상품 점수 + 공동출현 기반으로 2~4개짜리 꿀조합 생성."""
    _ensure_product_stats_built()
    _ensure_product_embeddings_built()

    _, products = _load_cu_products()
    user_emb = _embed_text(user_text)

    scored_products: List[Tuple[float, Dict[str, Any]]] = []

    for p in products:
        name = p["name"]

        if not prefs.allow_alcohol and any(
                kw in name for kw in ["맥주", "소주", "와인", "막걸리", "하이볼"]
        ):
            continue

        if "라면/분식" in prefs.banned_categories:
            if any(re.search(pat, name) for pat in _RAMEN_NAME_PATTERNS):
                continue

        if not _is_food_item(name, p["main_category"]):
            continue

        tags = _product_tags.get(name, set())
        prod_emb = None
        if _product_embeddings:
            prod_emb = _product_embeddings.get(name)

        s = _score_product_for_intent(p, tags, intent, prefs, user_emb, prod_emb)
        if s <= 0:
            continue
        scored_products.append((s, p))

    if not scored_products:
        return []

    scored_products.sort(key=lambda x: x[0], reverse=True)
    # 상위 상품 중에서 anchor 후보
    anchors = [p for _, p in scored_products[:40]]

    results: List[HoneyCombo] = []
    used_pairs: Set[Tuple[str, str]] = set()

    for idx, anchor in enumerate(anchors):
        if len(results) >= top_k:
            break

        anchor_name = anchor["name"]
        anchor_main_cat = anchor["main_category"]

        coocc_neighbors = _product_coocc.get(anchor_name, {})
        # coocc 기반으로 파트너 후보
        neighbor_names_sorted = sorted(
            coocc_neighbors.items(), key=lambda x: -x[1]
        )
        partners: List[Dict[str, Any]] = []

        for nb_name, _ in neighbor_names_sorted:
            if len(partners) >= 3:
                break
            nb_prod = next((p for p in products if p["name"] == nb_name), None)
            if not nb_prod:
                continue
            if not _is_food_item(nb_prod["name"], nb_prod["main_category"]):
                continue
            if "라면/분식" in prefs.banned_categories:
                if any(re.search(pat, nb_prod["name"]) for pat in _RAMEN_NAME_PATTERNS):
                    continue
            partners.append(nb_prod)

        # 공동출현이 거의 없으면, 점수순으로 추가
        if not partners:
            for _, p in scored_products:
                if p["name"] == anchor_name:
                    continue
                if not _is_food_item(p["name"], p["main_category"]):
                    continue
                if "라면/분식" in prefs.banned_categories:
                    if any(re.search(pat, p["name"]) for pat in _RAMEN_NAME_PATTERNS):
                        continue
                partners.append(p)
                if len(partners) >= 3:
                    break

        # anchor + partners 중에서 실제로 2~4개 선택
        # 식사 anchor면 디저트/사이드 섞기
        combo_products: List[Dict[str, Any]] = [anchor]

        for p in partners:
            if len(combo_products) >= 4:
                break
            # 중복 방지
            if any(cp["name"] == p["name"] for cp in combo_products):
                continue
            combo_products.append(p)

        if len(combo_products) < 2:
            continue

        # anchor-첫파트너 페어 중복 방지
        pair_key = tuple(sorted([combo_products[0]["name"], combo_products[1]["name"]]))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        category = _category_from_products(combo_products, prefs, intent)
        title = _build_combo_title_from_intent(user_text, intent, category)

        items: List[ComboItem] = []
        total_price = 0
        for p in combo_products:
            price = _normalize_price(p["price"])
            items.append(
                ComboItem(
                    original_name=p["name"],
                    name=p["name"],
                    price=price,
                    main_category=p["main_category"],
                )
            )
            if price:
                total_price += price

        combo = HoneyCombo(
            id=100000 + idx,
            name=title,
            category=category,
            items=items,
            total_price=_normalize_price(total_price),
            mood=None,
            generated=True,
        )

        combo = apply_negative_preferences_and_diet(combo, prefs)
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
    1) 사용자가 '밴쯔 정식', '앙버터 토스트'처럼 실제 조합 이름을 말한 경우
       → CSV에서 해당 조합을 찾아 그대로 반환
    2) 그 외 대부분의 경우
       → 사용자 입력 임베딩 + 상품 임베딩 + 태그 + 공동출현을 이용해
         CU 상품에서 실시간으로 꿀조합 패턴에 맞는 조합을 생성
    """
    # 1) 콤보 이름 직접 매칭 시도
    named_combo_row = _find_combo_by_name(user_text)
    if named_combo_row is not None:
        combo = _build_honey_combo_from_named_combo(named_combo_row, filters)
        if combo is not None:
            # 메인 콤보 + 비슷한 상품 기반 추가 조합들
            intent = analyze_user_intent(user_text)
            others = _build_combos_from_products(user_text, intent, filters, top_k=top_k)
            others = [c for c in others if c.name != combo.name][: max(0, top_k - 1)]
            return [combo] + others

    # 2) 일반 문장 → 상품 기반 실시간 조합
    intent = analyze_user_intent(user_text)
    combos = _build_combos_from_products(user_text, intent, filters, top_k=top_k)
    return combos


def generate_combos_product2vec(
        user_text: str,
        base_candidates: List[HoneyCombo],
        max_new: int,
        filters: UserPreferences,
) -> List[HoneyCombo]:
    """
    기존 시그니처 유지용.
    지금은 product2vec 대신,
    - base_candidates가 비어 있으면 recommend_combos_openai_rag 기반,
    - 비어 있지 않으면 base_candidates를 참고해
      상품 기반으로 추가 조합을 약간 섞어 주는 형태로 사용 가능.
    타임아웃을 고려해서, 별도의 OpenAI 호출은 하지 않는다
    (이미 recommend_combos_openai_rag에서 임베딩을 사용).
    """
    if max_new <= 0:
        return []

    # base_candidates를 참고해서 분위기 비슷한 조합을 하나 더 만들고 싶다면,
    # 여기서 간단히 _build_combos_from_products를 다시 호출해도 됨.
    # 중복 방지 정도만 해서 반환.
    intent = analyze_user_intent(user_text)
    extra = _build_combos_from_products(user_text, intent, filters, top_k=max_new + len(base_candidates))

    used_ids = {c.id for c in base_candidates}
    results: List[HoneyCombo] = []
    for c in extra:
        if c.id in used_ids:
            continue
        results.append(c)
        if len(results) >= max_new:
            break
    return results
