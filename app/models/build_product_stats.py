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
    """
    CU 상품명 / 콤보 상품명을 동일 규칙으로 정규화.
    - 괄호/특수문자 제거
    - 브랜드 접두어(샐, 면, 주, 도, 피치 등) 제거
    - 소문자 + 공백 축소
    """
    if not isinstance(name, str):
        name = str(name)
    s = name.lower()
    # 괄호 내용/기호 제거
    s = re.sub(r"[\(\)\[\]{}]", " ", s)
    s = re.sub(r"[^0-9a-z가-힣 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # 자주 등장하는 접두 브랜드 제거
    brands = ["샐", "면", "주", "도", "피치", "t1", "gs", "씨유"]
    for b in brands:
        s = s.replace(b.lower() + " ", "")
        s = s.replace(" " + b.lower(), "")
        if s.startswith(b.lower()):
            s = s[len(b):].strip()
    return s.strip()


def load_cu_products() -> Dict[str, str]:
    """
    CU 상품 CSV를 읽어서
    - 원래 이름: name
    - 정규화 이름: normalize_name(name)
    으로 딕셔너리 생성.
    """
    cu_df = pd.read_csv(CU_PRODUCTS_PATH)
    cu_norm_map: Dict[str, str] = {}

    for n in cu_df["name"]:
        original = str(n)
        normed = normalize_name(original)
        if not normed:
            continue
        # 같은 norm 키에 여러 개가 매핑될 수 있지만, 일단 최초 1개만 사용
        cu_norm_map.setdefault(normed, original)

    print(f"[build_product_stats] CU products = {len(cu_df)}, norm keys = {len(cu_norm_map)}")
    return cu_norm_map


def match_items_to_cu(items: List[str], cu_norm_map: Dict[str, str]) -> List[str]:
    """
    콤보 CSV에서 뽑은 raw item 리스트를
    CU 상품명(norm map)으로 매핑.
    """
    matched: List[str] = []
    for raw in items:
        normed = normalize_name(raw)
        if not normed:
            continue

        # 1차: 완전 일치
        if normed in cu_norm_map:
            name = cu_norm_map[normed]
            if name not in matched:
                matched.append(name)
            continue

        # 2차: 부분 문자열 매칭
        # ex) "불닭볶음면" ⊂ "오리지널불닭볶음면"
        candidates = [
            (k, v) for k, v in cu_norm_map.items()
            if normed in k or k in normed
        ]
        if candidates:
            # 그냥 첫 번째 후보 사용 (필요하면 더 정교하게 가능)
            _, name = candidates[0]
            if name not in matched:
                matched.append(name)
            continue

        # 3차: 못 맞추는 경우는 버림
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
        # '+', ',', '|' 기준으로 split
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
        if cand in row.index and not pd.isna[row.get(cand, None)]:
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

    # CU 상품 정규화 맵 로드
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

    # 상품별 태그 카운트, 상품별 등장 횟수
    product_tag_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    product_count: Dict[str, int] = defaultdict(int)

    # 상품간 co-occurrence 카운트
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

        # 상품별 태그 카운트 누적
        for name in matched_items:
            product_count[name] += 1
            for tag in tags:
                product_tag_count[name][tag] += 1

        # co-occurrence 카운트 (unordered pair)
        unique_items = sorted(set(matched_items))
        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                a = unique_items[i]
                b = unique_items[j]
                coocc[a][b] += 1
                coocc[b][a] += 1

    print(f"[build_product_stats] matched combo rows = {matched_combo_rows}")

    # product_tags.json 생성 (각 상품별 태그 리스트)
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

    # product_cooccurrence.json 생성 ({상품: {다른상품: count}})
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
