import os
import json
from collections import defaultdict
from typing import List, Dict, Set

import pandas as pd

# ---------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

COMB_PATH = os.path.join(DATA_DIR, "combination.csv")
SYN_PATH = os.path.join(DATA_DIR, "synthetic_honey_combos_1000.csv")
CU_PRODUCTS_PATH = os.path.join(DATA_DIR, "cu_official_products.csv")
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
# CSV 파서
# ---------------------------------------------------------

def load_combo_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] {path} 가 없습니다. 건너뜀.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def parse_items_from_row(row: pd.Series) -> List[str]:
    """
    콤보 CSV 한 줄에서 상품 리스트만 뽑기.
    기본 가정: 'items' 컬럼에 "상품1 + 상품2 + 상품3" 형태.
    없으면 'item', '상품'으로 시작하는 컬럼들을 모두 모음.
    """
    cols = row.index

    if "items" in cols:
        raw = str(row["items"]) if not pd.isna(row["items"]) else ""
        if not raw:
            return []
        # '+', ',', '|' 기준으로 대충 split
        for sep in ["+", ",", "|", "/"]:
            if sep in raw:
                parts = [p.strip() for p in raw.split(sep)]
                return [p for p in parts if p]
        return [raw.strip()] if raw.strip() else []

    # fallback: item1, item2, 상품1, 상품2 ...
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
    """
    콤보 이름/카테고리/설명 텍스트를 한 줄로 합치기.
    combo_name / name / category / 설명 등 여러 가능성을 고려.
    """
    pieces = []

    for cand in ["combo_name", "name", "콤보이름", "조합명"]:
        if cand in row.index and not pd.isna(row[cand]):
            pieces.append(str(row[cand]))

    for cand in ["category", "카테고리"]:
        if cand in row.index and not pd.isna(row[cand]):
            pieces.append(str(row[cand]))

    for cand in ["description", "설명", "comment", "메모", "mood"]:
        if cand in row.index and not pd.isna(row[cand]):
            pieces.append(str(row[cand]))

    # items 텍스트도 조금 보태기
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

    # 스트레스/우중충이면 위로 태그도 같이
    if TAG_STRESS in tags or TAG_RAINY in tags:
        tags.add(TAG_COMFORT)

    return tags


# ---------------------------------------------------------
# 메인 빌드 로직
# ---------------------------------------------------------

def main():
    print("[build_product_stats] 시작")

    cu_df = pd.read_csv(CU_PRODUCTS_PATH)
    cu_names: Set[str] = set(str(n) for n in cu_df["name"])

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

    # 상품별 태그 카운트, 상품별 등장 횟수
    product_tag_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    product_count: Dict[str, int] = defaultdict(int)

    # 상품간 co-occurrence 카운트
    coocc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for idx, row in df_all.iterrows():
        items = parse_items_from_row(row)
        items = [i for i in items if i in cu_names]  # CU에 실제 존재하는 상품만 사용
        if len(items) < 1:
            continue

        combo_text = extract_combo_text(row)
        tags = extract_tags_from_combo_text(combo_text)

        # 상품별 태그 카운트 누적
        for name in items:
            product_count[name] += 1
            for tag in tags:
                product_tag_count[name][tag] += 1

        # co-occurrence 카운트 (unordered pair)
        unique_items = sorted(set(items))
        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                a = unique_items[i]
                b = unique_items[j]
                coocc[a][b] += 1
                coocc[b][a] += 1

    # product_tags.json 생성 (각 상품별 태그 리스트)
    product_tags: Dict[str, List[str]] = {}
    for name, tag_dict in product_tag_count.items():
        total = product_count[name]
        tags_for_product: List[str] = []
        for tag, cnt in tag_dict.items():
            # 단순 기준: 한 번이라도 등장하면 태그로 부여 (원하면 cnt/total 비율로 threshold 줄 수도 있음)
            if cnt >= 1:
                tags_for_product.append(tag)
        product_tags[name] = sorted(tags_for_product)

    print(f"[build_product_stats] product_tags for {len(product_tags)} products")

    with open(PRODUCT_TAGS_PATH, "w", encoding="utf-8") as f:
        json.dump(product_tags, f, ensure_ascii=False, indent=2)

    # product_cooccurrence.json 생성 ({상품: {다른상품: count}})
    # 너무 크면 상위 몇 개만 자르고 싶으면 여기서 자를 수 있음 (예: count 상위 50개)
    trimmed_coocc: Dict[str, Dict[str, int]] = {}
    for name, neighbors in coocc.items():
        # count 기준 내림차순 정렬 후 상위 50개만
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: -x[1])[:50]
        trimmed_coocc[name] = {k: int(v) for k, v in sorted_neighbors}

    print(f"[build_product_stats] cooccurrence for {len(trimmed_coocc)} products")

    with open(COOCC_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed_coocc, f, ensure_ascii=False, indent=2)

    print("[build_product_stats] 완료")


if __name__ == "__main__":
    main()
