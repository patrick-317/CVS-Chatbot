import os
import re
import json
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from openai import OpenAI


# ============================================================
# 전역 캐시 / 상수
# ============================================================

_combo_docs: Optional[List[dict]] = None
_combo_embeddings: Optional[np.ndarray] = None
_openai_client: Optional[OpenAI] = None

EMBEDDING_MODEL = "text-embedding-3-small"
DATA_DIR = "data"
PRECOMP_DIR = "precomputed"


# ============================================================
# 공통 유틸
# ============================================================

def _json_default(o):
    """
    json.dump 에서 numpy 타입 등을 파이썬 기본 타입으로 변환하기 위한 헬퍼
    """
    import numpy as _np

    if isinstance(o, (_np.integer, _np.floating)):
        return o.item()
    return str(o)


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N, D), b: (D,) 또는 (1, D)
    return: (N,) similarity
    """
    if b.ndim == 1:
        b = b[None, :]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_norm @ b_norm.T).reshape(-1)


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    return pd.read_csv(path)


def _normalize_name(name: str) -> str:
    """
    편의점 상품명 매칭용 정규화:
    - 공백 제거
    - 소문자 변환
    - 숫자/영문/한글만 남김
    """
    s = str(name or "")
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-z가-힣]", "", s)
    return s


# ============================================================
# CU 상품 마스터 로딩
# ============================================================

def _prepare_product_master() -> pd.DataFrame:
    """
    CU 공식 상품 리스트 로드 + 정규화 컬럼 추가
    - data/cu_official_products.csv 사용
    """
    path = os.path.join(DATA_DIR, "cu_official_products.csv")
    df = _load_csv(path)

    # 이름 컬럼 추측
    if "name" in df.columns:
        name_col = "name"
    else:
        name_candidates = [
            c for c in df.columns
            if any(k in str(c).lower() for k in ["name", "상품명"])
        ]
        if not name_candidates:
            raise ValueError("cu_official_products.csv 에서 상품명 컬럼을 찾지 못했습니다.")
        name_col = name_candidates[0]

    # 가격 컬럼 추측
    if "price" in df.columns:
        price_col = "price"
    else:
        price_candidates = [
            c for c in df.columns
            if any(k in str(c).lower() for k in ["price", "가격"])
        ]
        if not price_candidates:
            raise ValueError("cu_official_products.csv 에서 가격 컬럼을 찾지 못했습니다.")
        price_col = price_candidates[0]

    df = df[[name_col, price_col]].copy()
    df.rename(columns={name_col: "name", price_col: "price"}, inplace=True)

    def _to_int_price(v) -> Optional[int]:
        if pd.isna(v):
            return None
        s = str(v).replace(",", "").strip()
        s = re.sub(r"[^0-9]", "", s)
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    df["price"] = df["price"].map(_to_int_price)
    df["name_norm"] = df["name"].map(_normalize_name)

    return df


def _match_item_name(target: str, df_products: pd.DataFrame) -> Tuple[Optional[str], Optional[int]]:
    """
    꿀조합에 적힌 상품명(target)을 CU 공식 상품명에 매칭
    - 우선: 정규화 후 완전 일치
    - 그 다음: 포함 관계
    - 마지막: fuzzy match (SequenceMatcher, threshold=0.6)
    """
    target_norm = _normalize_name(target)
    if not target_norm:
        return None, None

    prod_names = df_products["name"].tolist()
    prod_norms = df_products["name_norm"].tolist()
    prod_prices = df_products["price"].tolist()

    # 1) exact normalized match
    for name, norm, price in zip(prod_names, prod_norms, prod_prices):
        if norm == target_norm:
            return name, price

    # 2) substring 포함
    for name, norm, price in zip(prod_names, prod_norms, prod_prices):
        if target_norm in norm or norm in target_norm:
            return name, price

    # 3) fuzzy match
    best_i = None
    best_score = 0.0
    for i, norm in enumerate(prod_norms):
        score = SequenceMatcher(None, target_norm, norm).ratio()
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score >= 0.6:
        return prod_names[best_i], prod_prices[best_i]

    return None, None


# ============================================================
# 콤보 CSV → 문서 리스트 생성
# ============================================================

def _normalize_category(raw: str, default: str = "기타") -> str:
    s = str(raw or "").strip()
    if not s:
        return default

    if "라면" in s or "분식" in s:
        return "라면/분식"
    if "간편" in s or "식사" in s or "도시락" in s:
        return "식사류"
    if "디저트" in s or "dessert" in s.lower():
        return "디저트"
    if "안주" in s or "야식" in s:
        return "술안주/야식"

    return default


def _split_items(text: str) -> List[str]:
    """
    '콕콕콕 스파게티, 의성마늘후랑크, 모짜렐라 치즈' 같은 문자열을
    대충 상품명 단위로 나누기 위한 간단한 스플리터
    """
    if not isinstance(text, str):
        return []
    parts = re.split(r"[,\n]", text)
    items: List[str] = []
    for p in parts:
        name = p.strip()
        if name:
            items.append(name)
    return items


def _build_combo_docs_from_df(
        df: pd.DataFrame,
        df_products: pd.DataFrame,
        id_offset: int,
) -> List[dict]:
    """
    하나의 CSV(DataFrame)에서 꿀조합 리스트 추출.
    - '조합 이름', '주요 상품', '보조 상품(들)', '카테고리', '키워드 / 상황' 사용
    - 상품명은 CU 공식 상품과 최대한 매칭하여 name / price 를 채움.
    - CU에 없는 상품은 제외.
    - 최종적으로 **CU 상품이 2개 이상인 조합만** 사용.
    """
    docs: List[dict] = []

    for ridx, row in df.iterrows():
        combo_name = str(row.get("조합 이름", "")).strip()
        if not combo_name:
            continue

        raw_category = row.get("카테고리", "")
        category = _normalize_category(raw_category, default="기타")

        mood = str(row.get("키워드 / 상황", "")).strip()

        main_item = str(row.get("주요 상품", "")).strip()
        sub_items = str(row.get("보조 상품(들)", "")).strip()

        all_item_names: List[str] = []
        if main_item:
            all_item_names.append(main_item)
        all_item_names.extend(_split_items(sub_items))

        matched_items: List[dict] = []
        total_price = 0

        for nm in all_item_names:
            official_name, price = _match_item_name(nm, df_products)
            if not official_name:
                continue
            item = {
                "original_name": nm,
                "name": official_name,
                "price": price,
            }
            matched_items.append(item)
            if isinstance(price, int):
                total_price += price

        # CU에 매칭된 상품이 2개 미만이면 스킵
        if len(matched_items) < 2:
            continue

        item_names_for_text = ", ".join(i["name"] for i in matched_items)

        base_text = (
            f"꿀조합 이름: {combo_name}. "
            f"카테고리: {category}. "
            f"구성 상품: {item_names_for_text}. "
        )
        if mood:
            base_text += f"어울리는 상황/분위기: {mood}."

        doc = {
            "id": int(id_offset + ridx),
            "name": combo_name,
            "category": category,
            "items": matched_items,
            "total_price": int(total_price) if total_price > 0 else None,
            "mood": mood,
            "embedding_text": base_text,
        }
        docs.append(doc)

    return docs


def _build_combo_docs() -> List[dict]:
    """
    combination.csv + synthetic_honey_combos_1000.csv 를 모두 읽어서
    하나의 콤보 문서 리스트로 반환.
    """
    df_products = _prepare_product_master()
    docs: List[dict] = []

    # 1) 실제 꿀조합 100개
    comb_path = os.path.join(DATA_DIR, "combination.csv")
    if os.path.exists(comb_path):
        df_real = _load_csv(comb_path)
        docs.extend(_build_combo_docs_from_df(df_real, df_products, id_offset=0))

    # 2) synthetic 꿀조합 1000개
    syn_path = os.path.join(DATA_DIR, "synthetic_honey_combos_1000.csv")
    if os.path.exists(syn_path):
        df_syn = _load_csv(syn_path)
        offset = len(docs)
        docs.extend(_build_combo_docs_from_df(df_syn, df_products, id_offset=offset))

    return docs


# ============================================================
# 임베딩 인덱스 (precomputed 파일 + 캐시)
# ============================================================

def _load_semantic_index() -> Tuple[List[dict], np.ndarray]:
    """
    서버 런타임에서 호출:
    - precomputed 파일이 있으면 로드
    - 없으면 CSV에서 즉석 생성 + 임베딩 계산 후 저장
    """
    global _combo_docs, _combo_embeddings

    if _combo_docs is not None and _combo_embeddings is not None:
        return _combo_docs, _combo_embeddings

    os.makedirs(PRECOMP_DIR, exist_ok=True)
    docs_path = os.path.join(PRECOMP_DIR, "combo_docs.json")
    emb_path = os.path.join(PRECOMP_DIR, "combo_embeddings.npy")

    # 1) precomputed 가 있으면 그대로 로드
    if os.path.exists(docs_path) and os.path.exists(emb_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            _combo_docs = json.load(f)
        _combo_embeddings = np.load(emb_path)
        return _combo_docs, _combo_embeddings

    # 2) 없으면 CSV에서 즉석 생성
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
    embeds = np.array([d.embedding for d in resp.data], dtype=np.float32)

    _combo_docs = docs
    _combo_embeddings = embeds

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2, default=_json_default)
    np.save(emb_path, embeds)

    return _combo_docs, _combo_embeddings


# ============================================================
# 카테고리 추론 (키워드 + 다이어트 규칙)
# ============================================================

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "라면/분식": ["라면", "컵라면", "국물라면", "떡볶이", "분식", "우동", "튀김", "어묵"],
    "식사류": ["밥", "식사", "도시락", "김치찌개", "덮밥", "카레", "죽", "파스타", "볶음밥"],
    "간편식": ["삼각김밥", "주먹밥", "햄버거", "샌드위치", "핫도그", "토스트"],
    "디저트": [
        "디저트", "빵", "케이크", "쿠키", "초콜릿",
        "젤리", "아이스크림", "빙수", "달달", "달콤", "달다",
    ],
    "술안주/야식": [
        "맥주", "소주", "와인", "안주", "야식",
        "치킨", "족발", "포차", "편맥", "편의점맥주",
    ],
}


def infer_category_from_text(text: str) -> Optional[str]:
    """
    유저 자연어 문장에서 대략적인 카테고리 추론
    - 다이어트/든든 키워드를 우선 처리
    """
    text = (text or "").lower()

    # 규칙 1: 다이어트 관련 → 식사류
    if any(kw in text for kw in ["다이어트", "칼로리", "살찔", "살 안", "체중", "운동 후"]):
        return "식사류"

    # 규칙 2: 든든/배고파/출출 → 식사류
    if any(kw in text for kw in ["든든", "배고파", "배고픈", "출출"]):
        return "식사류"

    # 기본 키워드 매칭
    best_cat: Optional[str] = None
    best_score = 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text)
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat if best_score > 0 else None


def _apply_diet_hard_filter(user_text: str, docs: List[dict], indices: List[int]) -> List[int]:
    """
    유저가 다이어트 관련 발화를 했을 때,
    디저트/과자/빙수 + 라면/야식 카테고리를 최대한 제외.
    """
    text = (user_text or "").lower()
    diet_mode = any(kw in text for kw in ["다이어트", "칼로리", "살찔", "살 안", "체중", "운동 후"])
    if not diet_mode:
        return indices

    # 👉 다이어트 모드에서 피하고 싶은 카테고리
    bad_categories = ["라면/분식", "술안주/야식", "디저트"]

    # 👉 다이어트 모드에서 피하고 싶은 단어들 (고칼로리/야식 느낌)
    bad_words = [
        "빙수", "아이스크림", "케이크", "쿠키", "초콜릿", "초코",
        "젤리", "달달", "달콤", "디저트",
        "라면", "매운", "매콤", "치킨", "야식", "맥주", "소주",
    ]

    filtered: List[int] = []
    for i in indices:
        d = docs[i]

        # 1) 카테고리로 먼저 컷
        cat = str(d.get("category", ""))
        if cat in bad_categories:
            continue

        # 2) 텍스트 내용으로 한 번 더 컷
        content = (
                str(d.get("embedding_text", "")) + " "
                + str(d.get("mood", "")) + " "
                + str(d.get("name", ""))
        )
        if any(bw in content for bw in bad_words):
            continue

        filtered.append(i)

    # 전부 걸러졌으면 원래 리스트 유지 (응답이 비는 것 방지)
    return filtered if filtered else indices


# ============================================================
# 추천 API (카카오 컨트롤러에서 직접 호출)
# ============================================================

def recommend_combos_openai_rag(
        user_text: str,
        top_k: int = 3,
        min_items: int = 2,
) -> List[dict]:
    """
    - user_text 임베딩
    - 사전 계산된 combo_embeddings 와 cosine similarity
    - 카테고리 및 최소 상품 개수 조건을 고려해 상위 top_k 개 조합 리턴
    """
    docs, embeds = _load_semantic_index()
    if not docs:
        return []

    # 텍스트 기반 카테고리 추론
    inferred_cat = infer_category_from_text(user_text)

    # 1차 후보: 카테고리 + 최소 상품 개수 조건
    candidate_indices: List[int] = []
    for i, d in enumerate(docs):
        items = d.get("items", [])
        if not isinstance(items, list) or len(items) < min_items:
            continue

        if inferred_cat:
            if d.get("category") == inferred_cat:
                candidate_indices.append(i)
        else:
            candidate_indices.append(i)

    # 카테고리 기준으로 아무것도 없으면, 최소 상품 조건만으로 전체에서 검색
    if not candidate_indices:
        candidate_indices = [
            i for i, d in enumerate(docs)
            if isinstance(d.get("items", []), list)
               and len(d.get("items", [])) >= min_items
        ]

    if not candidate_indices:
        return []

    # 다이어트 모드 하드 필터 적용
    candidate_indices = _apply_diet_hard_filter(user_text, docs, candidate_indices)

    # user_text 임베딩
    client = _get_openai_client()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[user_text],
    )
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32)

    # 후보들에 대해서만 코사인 유사도 계산
    cand_embeds = embeds[candidate_indices]
    sims = _cosine_sim(cand_embeds, q_emb)

    # top_k * 3 정도 넉넉하게 뽑은 후 필터링
    top_n = min(len(candidate_indices), top_k * 3)
    order = np.argsort(-sims)[:top_n]

    results: List[dict] = []
    for ord_idx in order:
        doc_idx = candidate_indices[int(ord_idx)]
        d = docs[doc_idx]

        items = d.get("items", [])
        if not isinstance(items, list) or len(items) < min_items:
            continue

        results.append(
            {
                "id": d["id"],
                "name": d["name"],
                "category": d.get("category", "기타"),
                "items": items,
                "total_price": d.get("total_price"),
                "mood": d.get("mood", ""),
            }
        )

        if len(results) >= top_k:
            break

    return results
