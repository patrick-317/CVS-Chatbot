import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set

import pandas as pd

from app.schemas.recommendation_model import HoneyCombo, ComboItem


# ---------------------------------------------------------
# 경로 / 상수
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CU_PRODUCTS_PATH = os.path.join(DATA_DIR, "cu_official_products.csv")
COMB_PATH = os.path.join(DATA_DIR, "combination.csv")
SYN_PATH = os.path.join(DATA_DIR, "synthetic_honey_combos_1000.csv")

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
# 전역 캐시
# ---------------------------------------------------------

_cu_df: Optional[pd.DataFrame] = None
_cu_name_map: Optional[Dict[str, Dict[str, Any]]] = None

_combo_rows: Optional[List[Dict[str, Any]]] = None
_product_tags: Dict[str, Set[str]] = {}
_product_coocc: Dict[str, Dict[str, int]] = {}


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
# 사용자 입력 해석
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
    """발화에서 기분/맛/술/식사/다이어트 여부를 추출."""
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


def infer_category_from_text(user_text: str) -> Optional[str]:
    """발화에서 선호 카테고리(식사/라면/안주/디저트)를 추론."""
    t = user_text.strip()
    if re.search(r"식사|한끼|끼니|밥|도시락|주먹밥|샌드위치|점심|저녁|अ침", t):
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
# 콤보 데이터셋 기반 추천
# ---------------------------------------------------------

def _score_combo_for_intent(
        combo_row: Dict[str, Any],
        intent: Intent,
        prefs: UserPreferences,
        user_text: str,
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

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in _combo_rows:
        s = _score_combo_for_intent(row, intent, prefs, user_text)
        if s > 0:
            scored.append((s, row))

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
        scored = fallback

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)

    results: List[HoneyCombo] = []
    for _, row in scored:
        combo = _build_honey_combo_from_combo_row(row, user_text, intent, prefs)
        if combo is None:
            continue
        results.append(combo)
        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------
# 컨트롤러에서 사용하는 공개 함수
# ---------------------------------------------------------

def recommend_combos_openai_rag(
        user_text: str,
        top_k: int,
        filters: UserPreferences,
) -> List[HoneyCombo]:
    """이름은 그대로 두고, combo CSV 기반 추천 엔진을 래핑."""
    intent = analyze_user_intent(user_text)
    return recommend_combos_from_dataset(user_text, intent, filters, top_k=top_k)


def generate_combos_product2vec(
        user_text: str,
        base_candidates: List[HoneyCombo],
        max_new: int,
        filters: UserPreferences,
) -> List[HoneyCombo]:
    """현재는 product2vec 생성형을 사용하지 않고, API 호환만 유지."""
    return []
