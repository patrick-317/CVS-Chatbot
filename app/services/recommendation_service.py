import os
import re
import json
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from openai import OpenAI


# ---------- 전역 캐시 ----------

_combo_docs: Optional[List[dict]] = None
_combo_embeddings: Optional[np.ndarray] = None
_openai_client: Optional[OpenAI] = None


def _json_default(o):
    """
    json.dump 에서 numpy 타입 등을 파이썬 기본 타입으로 변환하기 위한 헬퍼
    """
    import numpy as np

    if isinstance(o, (np.integer, np.floating)):
        return o.item()  # numpy -> python int/float
    return str(o)


# ---------- OpenAI 클라이언트 ----------

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


EMBEDDING_MODEL = "text-embedding-3-small"


# ---------- 유틸 ----------

def _normalize_text(s: str) -> str:
    """이름 매칭용 간단 정규화 (공백/특수문자 제거 + 소문자)."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = re.sub(r"[\s\(\)\[\]\{\}\/\-\+·…·.,'\"!?]", "", s)
    return s.lower()


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value)
    # 숫자/콤마 외 제거
    s = re.sub(r"[^0-9]", "", s)
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------- 데이터 로딩 ----------

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path)
    return df


def _prepare_product_master() -> pd.DataFrame:
    """
    CU 공식 상품 리스트 로드 + 정규화 컬럼 추가
    - data/cu_official_products.csv 를 사용
    - name / NAME / 상품명 등 이름 컬럼, price / PRICE / 가격 등 가격 컬럼 자동 탐색
    """
    df = _load_csv("data/cu_official_products.csv")

    # 이름 컬럼 찾기
    name_cols = [
        c for c in df.columns
        if any(k in str(c) for k in ["상품명", "name", "NAME"])
    ]
    if not name_cols:
        raise ValueError("cu_official_products.csv 에서 상품명 컬럼을 찾지 못했습니다.")
    name_col = name_cols[0]

    # 가격 컬럼 찾기
    price_cols = [
        c for c in df.columns
        if any(k in str(c) for k in ["가격", "price", "PRICE"])
    ]
    if not price_cols:
        raise ValueError("cu_official_products.csv 에서 가격 컬럼을 찾지 못했습니다.")
    price_col = price_cols[0]

    df = df[[name_col, price_col]].copy()
    df.rename(columns={name_col: "name", price_col: "price"}, inplace=True)

    df["name"] = df["name"].astype(str)
    df["name_norm"] = df["name"].apply(_normalize_text)
    df["price_int"] = df["price"].apply(_safe_int)

    return df


def _guess_combo_name_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if any(k in str(c) for k in ["조합명", "꿀조합", "세트명", "세트", "이름", "name", "title"])
    ]
    return candidates[0] if candidates else None


def _guess_item_columns(df: pd.DataFrame) -> List[str]:
    """
    주상품/보조상품, item1, product1 같은 컬럼들을 모두 아이템 컬럼으로 사용
    """
    cols = []
    for c in df.columns:
        cs = str(c)
        if any(k in cs for k in ["주상품", "보조상품", "상품", "item", "Item", "product", "Product"]):
            cols.append(c)
    return cols


def _guess_category_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if any(k in str(c) for k in ["카테고리", "category", "분류"])
    ]
    return candidates[0] if candidates else None


def _guess_mood_column(df: pd.DataFrame) -> Optional[str]:
    # 기분/상황 태그 등
    candidates = [
        c for c in df.columns
        if any(k in str(c) for k in ["상황", "키워드", "분위기", "태그", "vibe", "mood"])
    ]
    return candidates[0] if candidates else None


# ---------- CU 상품 매칭 (여기서 CU에 없는 상품은 버린다) ----------

def _build_combo_docs_from_df(
        df: pd.DataFrame,
        df_products: pd.DataFrame,
        default_category: str,
        id_offset: int,
) -> List[dict]:
    """
    하나의 CSV(DataFrame)에서 꿀조합 리스트 추출.
    - 상품명은 CU 공식 상품과 최대한 매칭하여 official_name / price 를 채움.
    - CU에 존재하지 않는 상품(매칭 실패)은 해당 콤보에서 제외.
    - 콤보 안의 모든 상품이 매칭 실패하면 그 콤보 자체를 버림.
    """
    combo_name_col = _guess_combo_name_column(df)
    item_cols = _guess_item_columns(df)
    category_col = _guess_category_column(df)
    mood_col = _guess_mood_column(df)

    combos: List[dict] = []

    if not item_cols:
        # 아이템 컬럼이 하나도 없으면 스킵
        return combos

    # 미리 product master 를 numpy 배열로 만들어 매칭 속도 약간 개선
    prod_names_norm = df_products["name_norm"].to_numpy()
    prod_names = df_products["name"].to_numpy()
    prod_prices = df_products["price_int"].to_numpy()

    def _match_product(item_name: str) -> Tuple[Optional[str], Optional[int]]:
        """
        item_name 을 CU 마스터와 매칭
        - 1순위: 정규화 이름 완전 일치
        - 2순위: fuzzy match (SequenceMatcher), 0.6 이상일 때만 채택
        - 매칭 실패 또는 가격 정보 없으면 (None, None) 반환
        """
        if not isinstance(item_name, str):
            item_name_str = str(item_name) if item_name is not None else ""
        else:
            item_name_str = item_name

        target_norm = _normalize_text(item_name_str)
        if not target_norm:
            return None, None

        # 1) exact match
        exact_idx = np.where(prod_names_norm == target_norm)[0]
        if len(exact_idx) > 0:
            i = int(exact_idx[0])
            price_val = prod_prices[i]
            price_int = _safe_int(price_val)
            if price_int is None:
                return None, None
            return prod_names[i], price_int

        # 2) fuzzy
        best_i = None
        best_score = 0.0
        for i, pn in enumerate(prod_names_norm):
            score = SequenceMatcher(None, target_norm, pn).ratio()
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is not None and best_score >= 0.6:
            price_val = prod_prices[best_i]
            price_int = _safe_int(price_val)
            if price_int is None:
                return None, None
            return prod_names[best_i], price_int

        # CU에 없는 상품으로 판단 → 추천에서 제외
        return None, None

    for ridx, row in df.iterrows():
        raw_name = (
            str(row[combo_name_col]).strip()
            if combo_name_col and pd.notna(row.get(combo_name_col))
            else ""
        )
        combo_name = raw_name or f"꿀조합 {id_offset + ridx}"

        # 카테고리
        if category_col and pd.notna(row.get(category_col)):
            category = str(row[category_col]).strip()
        else:
            category = default_category

        # 아이템들(원본)
        items_raw: List[str] = []
        for c in item_cols:
            val = row.get(c)
            if pd.isna(val):
                continue
            s = str(val).strip()
            if not s:
                continue
            items_raw.append(s)

        if not items_raw:
            continue

        # CU 공식 상품으로 매핑된 아이템만 모으기
        items: List[dict] = []
        for item_name in items_raw:
            official_name, price_int = _match_product(item_name)
            # CU에 없는 상품은 통째로 버린다
            if official_name is None or price_int is None:
                continue

            items.append(
                {
                    "original_name": item_name,
                    "name": official_name,  # ✅ 반드시 CU 상품명
                    "price": int(price_int),
                }
            )

        # CU 상품이 하나도 없는 콤보는 버림
        if not items:
            continue

        # 총 가격 계산 (모든 상품에 price가 있다고 가정)
        total_price = sum(i["price"] for i in items)

        # 분위기/상황 키워드
        mood = ""
        if mood_col and pd.notna(row.get(mood_col)):
            mood = str(row[mood_col]).strip()

        # embedding 에 사용할 텍스트 (CU 공식 상품명 기준)
        item_names_for_text = ", ".join(i["name"] for i in items)
        base_text = (
            f"꿀조합 이름: {combo_name}. "
            f"카테고리: {category}. "
            f"구성 상품: {item_names_for_text}. "
        )
        if mood:
            base_text += f"어울리는 상황/분위기: {mood}."

        doc = {
            "id": id_offset + ridx,
            "name": combo_name,
            "category": category,
            "items": items,               # ✅ 전부 CU 상품
            "total_price": total_price,   # ✅ 항상 int
            "mood": mood,
            "embedding_text": base_text,
        }
        combos.append(doc)

    return combos


def _build_combo_docs() -> List[dict]:
    """
    combination.csv + synthetic_honey_combos_1000.csv 를 통합해서
    하나의 꿀조합 리스트로 만든다.
    """
    df_products = _prepare_product_master()

    docs: List[dict] = []

    # 1) combination.csv (실제 꿀조합)
    try:
        df_combo = _load_csv("data/combination.csv")
        docs += _build_combo_docs_from_df(
            df_combo,
            df_products=df_products,
            default_category="기타",
            id_offset=0,
        )
    except FileNotFoundError:
        pass

    # 2) synthetic_honey_combos_1000.csv (synthetic 데이터)
    try:
        df_syn = _load_csv("data/synthetic_honey_combos_1000.csv")
        docs += _build_combo_docs_from_df(
            df_syn,
            df_products=df_products,
            default_category="기타",
            id_offset=10_000,
        )
    except FileNotFoundError:
        pass

    return docs


# ---------- 임베딩 사전 계산 ----------

def build_precomputed_embeddings():
    """
    - data/*.csv 를 읽어서 combo_docs.json + combo_embeddings.npy 생성
    - 개발/배포 시 한 번 실행
    """
    print("[build_precomputed_embeddings] 콤보 문서 생성 중...")
    docs = _build_combo_docs()
    print(f"[build_precomputed_embeddings] 총 {len(docs)} 개 조합 처리 중...")

    if not docs:
        print("[build_precomputed_embeddings] 생성된 콤보가 없습니다. CSV 컬럼 매핑을 확인하세요.")
        return

    client = _get_openai_client()

    texts = [d["embedding_text"] for d in docs]
    embeddings: List[List[float]] = []

    # 너무 길어지는 것 방지용 batch
    BATCH = 100
    for i in range(0, len(texts), BATCH):
        batch = texts[i: i + BATCH]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for e in resp.data:
            embeddings.append(e.embedding)

    arr = np.array(embeddings, dtype=np.float32)

    os.makedirs("precomputed", exist_ok=True)
    with open("precomputed/combo_docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2, default=_json_default)
    np.save("precomputed/combo_embeddings.npy", arr)

    print("[build_precomputed_embeddings] 저장 완료: precomputed/combo_docs.json, combo_embeddings.npy")


def _load_semantic_index() -> Tuple[List[dict], np.ndarray]:
    """
    서버 런타임에서 호출:
    - precomputed 존재하면 바로 로드
    - 없으면 즉석에서 생성(속도 느릴 수 있음)
    """
    global _combo_docs, _combo_embeddings

    if _combo_docs is not None and _combo_embeddings is not None:
        return _combo_docs, _combo_embeddings

    docs_path = "precomputed/combo_docs.json"
    emb_path = "precomputed/combo_embeddings.npy"

    if os.path.exists(docs_path) and os.path.exists(emb_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            _combo_docs = json.load(f)
        _combo_embeddings = np.load(emb_path)
        return _combo_docs, _combo_embeddings

    # fallback: 즉석 생성
    print("[_load_semantic_index] precomputed 파일이 없어, CSV에서 즉석 생성합니다.")
    docs = _build_combo_docs()
    if not docs:
        raise RuntimeError("콤보 데이터를 하나도 만들지 못했습니다. CSV 구조를 확인하세요.")

    client = _get_openai_client()
    texts = [d["embedding_text"] for d in docs]
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    embeds = np.array([e.embedding for e in resp.data], dtype=np.float32)

    _combo_docs = docs
    _combo_embeddings = embeds
    return _combo_docs, _combo_embeddings


# ---------- 추천 로직 (RAG) ----------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, keepdims=True) + 1e-8)
    return a_norm @ b_norm


def recommend_combos_openai_rag(user_text: str, top_k: int = 3) -> List[dict]:
    """
    - user_text 임베딩
    - 사전 계산된 combo_embeddings 와 cosine similarity
    - 상위 top_k 개 조합 리턴
    """
    docs, embeds = _load_semantic_index()
    if not docs:
        return []

    client = _get_openai_client()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[user_text],
    )
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32)

    sims = _cosine_sim(embeds, q_emb)
    idx_sorted = np.argsort(-sims)[: top_k]

    results: List[dict] = []
    for idx in idx_sorted:
        d = docs[int(idx)]
        results.append(
            {
                "id": d["id"],
                "name": d["name"],
                "category": d.get("category", "기타"),
                "items": d.get("items", []),         # ✅ 이미 CU 상품만
                "total_price": d.get("total_price"), # ✅ int
                "mood": d.get("mood", ""),
            }
        )

    return results


# ---------- (선택) 카테고리 추론 ----------

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "라면/분식": ["라면", "컵라면", "국물라면", "떡볶이", "분식", "우동", "튀김", "어묵"],
    "식사류": ["밥", "식사", "도시락", "김치찌개", "덮밥", "카레", "죽", "파스타", "볶음밥"],
    "간편식": ["삼각김밥", "주먹밥", "햄버거", "샌드위치", "핫도그", "토스트"],
    "디저트": [
        "디저트",
        "빵",
        "케이크",
        "쿠키",
        "초콜릿",
        "젤리",
        "아이스크림",
        "빙수",
        "달달",
        "달콤",
        "달다",
    ],
    "술안주/야식": [
        "맥주",
        "소주",
        "와인",
        "안주",
        "야식",
        "치킨",
        "족발",
        "포차",
        "편맥",
        "편의점맥주",
    ],
}


def infer_category_from_text(text: str) -> Optional[str]:
    text = text or ""
    best_cat = None
    best_score = 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat if best_score > 0 else None
