from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

# ------------------------
# 전역 설정 & 캐시
# ------------------------

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
PRECOMP_DIR = BASE_DIR / "precomputed"
PRECOMP_DIR.mkdir(exist_ok=True)

COMBO_CSV_CANDIDATES = [
    DATA_DIR / "combination.csv",
    BASE_DIR / "combination.csv",
    ]
SYN_CSV_CANDIDATES = [
    DATA_DIR / "synthetic_honey_combos_1000.csv",
    BASE_DIR / "synthetic_honey_combos_1000.csv",
    ]
PRODUCT_CSV_CANDIDATES = [
    DATA_DIR / "cu_official_products.csv",
    BASE_DIR / "cu_official_products.csv",
    ]

DOCS_JSON_PATH = PRECOMP_DIR / "combo_docs.json"
EMBED_NPY_PATH = PRECOMP_DIR / "combo_embeddings.npy"

_openai_client: Optional[OpenAI] = None
_combo_docs: Optional[List[Dict]] = None
_combo_embeddings: Optional[np.ndarray] = None
_df_combo: Optional[pd.DataFrame] = None
_df_syn: Optional[pd.DataFrame] = None
_df_products: Optional[pd.DataFrame] = None


# ------------------------
# 유틸
# ------------------------

def _resolve_first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.is_file():
            return p
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {[str(p) for p in paths]}")


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _parse_price(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None


def _load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global _df_combo, _df_syn, _df_products

    if _df_combo is not None and _df_syn is not None and _df_products is not None:
        return _df_combo, _df_syn, _df_products

    combo_path = _resolve_first_existing(COMBO_CSV_CANDIDATES)
    syn_path = _resolve_first_existing(SYN_CSV_CANDIDATES)
    prod_path = _resolve_first_existing(PRODUCT_CSV_CANDIDATES)

    _df_combo = pd.read_csv(combo_path, encoding="utf-8-sig")
    _df_syn = pd.read_csv(syn_path, encoding="utf-8-sig")
    _df_products = pd.read_csv(prod_path, encoding="utf-8-sig")

    # 가격 정규화
    _df_products["price_int"] = _df_products["price"].apply(_parse_price)

    return _df_combo, _df_syn, _df_products


def _find_price(product_name: str, df_products: pd.DataFrame) -> Optional[int]:
    if not product_name:
        return None

    # 1차: 완전 일치
    exact = df_products.loc[df_products["name"] == product_name]
    if not exact.empty:
        return _parse_price(exact.iloc[0]["price"])

    # 2차: 부분 일치 (regex 사용 X)
    contains = df_products[
        df_products["name"].str.contains(product_name, na=False, regex=False)
    ]
    if not contains.empty:
        return _parse_price(contains.iloc[0]["price"])

    return None



# ------------------------
# 콤보 문서 구축
# ------------------------

def _build_combo_docs() -> List[Dict]:
    df_combo, df_syn, df_products = _load_dataframes()

    # 원본 + synthetic 합치기
    df_all = pd.concat([df_combo, df_syn], ignore_index=True)

    docs: List[Dict] = []
    for idx, row in df_all.iterrows():
        name = str(row.get("조합 이름", "")).strip() or f"꿀조합 {idx}"
        store = str(row.get("편의점", "")).strip()
        main_item = str(row.get("주요 상품", "")).strip()
        extra_items_raw = str(row.get("보조 상품(들)", "")).strip()
        keywords = str(row.get("키워드 / 상황", "")).strip()
        category = str(row.get("카테고리", "")).strip() or "기타"

        # 보조 상품 파싱 (쉼표 기준)
        extra_items: List[str] = []
        if extra_items_raw:
            extra_items = [x.strip() for x in extra_items_raw.split(",") if x.strip()]

        all_item_names: List[str] = []
        if main_item:
            all_item_names.append(main_item)
        all_item_names.extend(extra_items)

        item_entries: List[Dict] = []
        total_price: Optional[int] = 0
        any_price = False

        for pname in all_item_names:
            price = _find_price(pname, df_products)
            if price is not None:
                any_price = True
                total_price = (total_price or 0) + price
            item_entries.append(
                {
                    "name": pname,
                    "price": price,
                }
            )

        if not any_price:
            total_price = None

        doc_text_parts = [
            name,
            f"카테고리: {category}",
            f"상황: {keywords}" if keywords else "",
            f"주요 상품: {main_item}" if main_item else "",
            f"보조 상품: {extra_items_raw}" if extra_items_raw else "",
        ]
        doc_text = " / ".join(part for part in doc_text_parts if part)

        docs.append(
            {
                "id": int(idx),
                "name": name,
                "store": store,
                "category": category,
                "keywords": keywords,
                "main_item": main_item,
                "extra_items": extra_items,
                "items": item_entries,      # ← 상품명 + 개별 가격
                "total_price": total_price,  # ← 총합 가격
                "doc_text": doc_text,        # ← 임베딩용 텍스트
            }
        )

    return docs


# ------------------------
# 임베딩 사전 계산
# ------------------------

def build_precomputed_embeddings() -> None:
    """
    combo_docs.json + combo_embeddings.npy 생성용 스크립트 함수.
    """
    global _combo_docs, _combo_embeddings

    client = _get_openai_client()
    docs = _build_combo_docs()
    print(f"[build_precomputed_embeddings] 총 {len(docs)} 개 조합 처리 중...")

    texts = [d["doc_text"] for d in docs]
    embeddings: List[List[float]] = []

    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        for e in resp.data:
            embeddings.append(e.embedding)

    _combo_docs = docs
    _combo_embeddings = np.array(embeddings, dtype="float32")

    PRECOMP_DIR.mkdir(exist_ok=True)
    with open(DOCS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    np.save(EMBED_NPY_PATH, _combo_embeddings)

    print("[build_precomputed_embeddings] 저장 완료:", DOCS_JSON_PATH, EMBED_NPY_PATH)


# ------------------------
# 런타임 로더
# ------------------------

def _load_semantic_index() -> Tuple[List[Dict], np.ndarray]:
    global _combo_docs, _combo_embeddings

    if _combo_docs is not None and _combo_embeddings is not None:
        return _combo_docs, _combo_embeddings

    if DOCS_JSON_PATH.is_file() and EMBED_NPY_PATH.is_file():
        with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
            _combo_docs = json.load(f)
        _combo_embeddings = np.load(EMBED_NPY_PATH)
        return _combo_docs, _combo_embeddings

    # 사전 계산 파일이 없으면 즉석 생성
    build_precomputed_embeddings()
    return _combo_docs, _combo_embeddings


# ------------------------
# 카테고리 키워드 매핑 (간단 버전)
# ------------------------

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
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


# ------------------------
# 추천 메인 함수
# ------------------------

def recommend_combos_openai_rag(user_text: str, top_k: int = 3) -> List[Dict]:
    """
    user_text 를 임베딩해서 combo_docs 와의 코사인 유사도로 top_k 추천.
    """
    docs, combo_embeds = _load_semantic_index()
    if not docs or combo_embeds is None or len(combo_embeds) == 0:
        return []

    client = _get_openai_client()
    q_resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[user_text],
    )
    q_vec = np.array(q_resp.data[0].embedding, dtype="float32")

    # 코사인 유사도
    doc_norms = np.linalg.norm(combo_embeds, axis=1, keepdims=True)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0 or np.any(doc_norms == 0):
        sims = combo_embeds @ q_vec
    else:
        sims = (combo_embeds @ q_vec) / (doc_norms.flatten() * q_norm)

    top_k = max(1, min(top_k, len(docs)))
    top_indices = np.argsort(-sims)[:top_k]

    results: List[Dict] = []
    for idx in top_indices:
        doc = docs[int(idx)]
        total_price = doc.get("total_price")

        price_line = ""
        if isinstance(total_price, (int, float)):
            price_line = f"이 조합을 모두 담으면 대략 {total_price:,}원 정도예요."

        reason_lines = [
            f"입력하신 문장의 의미를 임베딩으로 분석해서 가장 비슷한 분위기의 꿀조합을 골랐어요. (기준: '{user_text}')",
        ]
        if doc.get("keywords"):
            reason_lines.append(f"이 조합은 '{doc['keywords']}' 상황에 잘 어울려요.")
        if price_line:
            reason_lines.append(price_line)

        result = {
            "name": doc.get("name", "편의점 꿀조합"),
            "category": doc.get("category", "기타"),
            "items": doc.get("items", []),
            "total_price": total_price,
            "reason": "\n\n".join(reason_lines),
        }
        results.append(result)

    return results
