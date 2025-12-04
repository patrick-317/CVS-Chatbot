import os
import json
import re
from collections import defaultdict
from typing import List, Dict, Set

import pandas as pd

# ---------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed")

COMB_PATH = os.path.join(DATA_DIR, "combination.csv")
SYN_PATH = os.path.join(DATA_DIR, "synthetic_honey_combos_1000.csv")
CU_PRODUCTS_PATH = os.path.join(DATA_DIR, "cu_official_products.csv")

PRODUCT_TAGS_PATH = os.path.join(PRECOMPUTED_DIR, "product_tags.json")
COOCC_PATH = os.path.join(PRECOMPUTED_DIR, "product_cooccurrence.json")

# ---------------------------------------------------------
# 태그 정의 (콤보 기반)
# ---------------------------------------------------------

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

_SPICY_KEYWORDS = [
    "불닭", "매운", "매콤", "마라", "핫", "청양", "화끈", "얼얼", "지옥", "스파이시"
]

_SWEET_KEYWORDS = [
    "초코", "초콜릿", "케익", "케이크", "쿠키", "브라우니", "마카롱", "젤리",
    "롤케이크", "롤케익", "파이", "타르트", "카라멜", "바닐라", "딸기", "바나나", "허니", "꿀"
]

_HOT_SOUP_KEYWORDS = [
    "탕", "국", "찌개", "국물", "라멘", "라면", "우동", "칼국수", "어묵", "오뎅"
]

_ALCOHOL_KEYWORDS = [
    "소주", "맥주", "막걸리", "와인", "하이볼", "RTD", "칵테일"
]

_PROTEIN_KEYWORDS = [
    "닭가슴살", "닭가슴", "닭 안심", "계란", "란", "프로틴", "단백질",
    "그릭요거트", "요거트", "두부", "콩", "참치", "연어", "고등어",
    "햄", "소시지", "소세지", "치즈"
]

_MEAL_KEYWORDS = [
    "도시락", "김밥", "주먹밥", "덮밥", "볶음밥", "비빔밥",
    "파스타", "라면", "면", "우동", "짜장", "짬뽕", "샌드위치", "버거"
]

_SNACK_KEYWORDS = [
    "과자", "스낵", "칩", "쿠키", "츄러스", "팝콘", "크래커", "젤리"
]

_DESSERT_KEYWORDS = _SWEET_KEYWORDS + [
    "디저트", "아이스크림", "빙수", "콘", "바"
]

_STRESS_KEYWORDS = [
    "스트레스", "폭발", "야근", "철야", "멘붕", "빡침", "현타"
]

_RAINY_KEYWORDS = [
    "비도 오고", "비 와", "비와", "비 오", "비오는", "비 오는", "우중충"
]

# ---------------------------------------------------------
# 이름 정규화 / 매칭
# ---------------------------------------------------------

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    s = name.lower()
    s = re.sub(r"[\(\)\[\]{}]", " ", s)
    s = re.sub(r"[^0-9a-z가-힣 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    brands = ["샐", "면", "주", "도", "피치", "t1", "gs", "씨유"]
    for b in brands:
        s = s.replace(b.lower() + " ", "")
        s = s.replace(" " + b.lower(), "")
        if s.startswith(b.lower()):
            s = s[len(b):].strip()
    return s.strip()


def load_cu_products() -> Dict[str, str]:
    cu_df = pd.read_csv(CU_PRODUCTS_PATH)
    cu_norm_map: Dict[str, str] = {}

    for n in cu_df["name"]:
        original = str(n)
        normed = normalize_name(original)
        if not normed:
            continue
        cu_norm_map.setdefault(normed, original)

    print(f"[build_product_stats] CU products = {len(cu_df)}, norm keys = {len(cu_norm_map)}")
    return cu_norm_map


def match_items_to_cu(items: List[str], cu_norm_map: Dict[str, str]) -> List[str]:
    matched: List[str] = []
    for raw in items:
        normed = normalize_name(raw)
        if not normed:
            continue

        if normed in cu_norm_map:
            name = cu_norm_map[normed]
            if name not in matched:
                matched.append(name)
            continue

        candidates = [
            (k, v) for k, v in cu_norm_map.items()
            if normed in k or k in normed
        ]
        if candidates:
            _, name = candidates[0]
            if name not in matched:
                matched.append(name)
            continue

    return matched

# ---------------------------------------------------------
# CSV 파서
# ---------------------------------------------------------

def load_combo_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] {path} 가 없습니다. 건너뜀.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def parse_items_from_row(row: pd.Series) -> List[str]:
    cols = row.index

    if "items" in cols:
        raw = str(row["items"]) if not pd.isna(row["items"]) else ""
        if not raw:
            return []
        for sep in ["+", ",", "|", "/"]:
            if sep in raw:
                parts = [p.strip() for p in raw.split(sep)]
                return [p for p in parts if p]
        return [raw.strip()] if raw.strip() else []


    item_names: List[str] = []
    for c in cols:
        cl = str(c).lower()
        if cl.startswith("item") or "상품" in cl:
            val = row[c]
            if pd.isna(val):
                continue
            name = str(val).strip()
            if name:
                item_names.append(name)
    return item_names


def extract_combo_text(row: pd.Series) -> str:
    pieces = []

    for cand in ["combo_name", "name", "콤보이름", "조합명"]:
        if cand in row.index and not pd.isna(row[cand]):
            pieces.append(str(row[cand]))

    for cand in ["category", "카테고리"]:
        if cand in row.index and not pd.isna(row[cand]):
            pieces.append(str(row[cand]))

    for cand in ["description", "설명", "comment", "메모", "mood"]:
        if cand in row.index and not pd.isna[row.get(cand, None)]:
            pieces.append(str(row[cand]))

    items = parse_items_from_row(row)
    if items:
        pieces.append(" + ".join(items))

    return " ".join(pieces)


def extract_tags_from_combo_text(text: str) -> Set[str]:
    t = text or ""
    tags: Set[str] = set()

    if any(kw in t for kw in _SPICY_KEYWORDS):
        tags.add(TAG_SPICY)
    if any(kw in t for kw in _SWEET_KEYWORDS):
        tags.add(TAG_SWEET)
    if any(kw in t for kw in _HOT_SOUP_KEYWORDS):
        tags.update([TAG_HOT_SOUP, TAG_COMFORT])
    if any(kw in t for kw in _ALCOHOL_KEYWORDS):
        tags.add(TAG_ALCOHOL)
    if any(kw in t for kw in _SNACK_KEYWORDS):
        tags.add(TAG_SNACK)
    if any(kw in t for kw in _DESSERT_KEYWORDS):
        tags.add(TAG_DESSERT)
    if any(kw in t for kw in _MEAL_KEYWORDS):
        tags.add(TAG_MEAL)
    if any(kw in t for kw in _PROTEIN_KEYWORDS):
        tags.add(TAG_PROTEIN)
    if any(kw in t for kw in _STRESS_KEYWORDS):
        tags.add(TAG_STRESS)
    if any(kw in t for kw in _RAINY_KEYWORDS):
        tags.add(TAG_RAINY)

    if TAG_STRESS in tags or TAG_RAINY in tags:
        tags.add(TAG_COMFORT)

    return tags

# ---------------------------------------------------------
# 메인 빌드 로직
# ---------------------------------------------------------

def main():
    print("[build_product_stats] 시작")

    cu_norm_map = load_cu_products()

    df_comb = load_combo_df(COMB_PATH)
    df_syn = load_combo_df(SYN_PATH)

    all_rows = []
    if not df_comb.empty:
        all_rows.append(df_comb)
    if not df_syn.empty:
        all_rows.append(df_syn)

    if not all_rows:
        print("[build_product_stats] 콤보 CSV가 없습니다. 종료.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    print(f"[build_product_stats] combo rows = {len(df_all)}")

    product_tag_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    product_count: Dict[str, int] = defaultdict(int)

    coocc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    matched_combo_rows = 0

    for idx, row in df_all.iterrows():
        raw_items = parse_items_from_row(row)
        matched_items = match_items_to_cu(raw_items, cu_norm_map)

        if len(matched_items) < 1:
            continue

        matched_combo_rows += 1

        combo_text = extract_combo_text(row)
        tags = extract_tags_from_combo_text(combo_text)

        for name in matched_items:
            product_count[name] += 1
            for tag in tags:
                product_tag_count[name][tag] += 1

        unique_items = sorted(set(matched_items))
        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                a = unique_items[i]
                b = unique_items[j]
                coocc[a][b] += 1
                coocc[b][a] += 1

    print(f"[build_product_stats] matched combo rows = {matched_combo_rows}")

    product_tags: Dict[str, List[str]] = {}
    for name, tag_dict in product_tag_count.items():
        tags_for_product: List[str] = []
        for tag, cnt in tag_dict.items():
            if cnt >= 1:
                tags_for_product.append(tag)
        product_tags[name] = sorted(tags_for_product)

    print(f"[build_product_stats] product_tags for {len(product_tags)} products")

    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    with open(PRODUCT_TAGS_PATH, "w", encoding="utf-8") as f:
        json.dump(product_tags, f, ensure_ascii=False, indent=2)

    trimmed_coocc: Dict[str, Dict[str, int]] = {}
    for name, neighbors in coocc.items():
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: -x[1])[:50]
        trimmed_coocc[name] = {k: int(v) for k, v in sorted_neighbors}

    print(f"[build_product_stats] cooccurrence for {len(trimmed_coocc)} products")

    with open(COOCC_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed_coocc, f, ensure_ascii=False, indent=2)

    print("[build_product_stats] 완료")


if __name__ == "__main__":
    main()
