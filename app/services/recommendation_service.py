import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

# =========================
# 경로 / 전역 변수
# =========================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
PRECOMPUTED_DIR = BASE_DIR / "precomputed"
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

COMBINATION_CSV = DATA_DIR / "combination.csv"
SYNTHETIC_CSV = DATA_DIR / "synthetic_honey_combos_1000.csv"
PRODUCTS_CSV = DATA_DIR / "cu_official_products.csv"

PRECOMP_DOCS_JSON = PRECOMPUTED_DIR / "combo_docs.json"
PRECOMP_EMB_NPY = PRECOMPUTED_DIR / "combo_embeddings.npy"

EMBED_MODEL = "text-embedding-3-small"

_openai_client: Optional[OpenAI] = None
_combo_docs: Optional[List[Dict[str, Any]]] = None
_combo_embeddings: Optional[np.ndarray] = None
_products_df: Optional[pd.DataFrame] = None

# =========================
# OpenAI 클라이언트
# =========================


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        # OPENAI_API_KEY 는 .env / 환경변수에 세팅되어 있다고 가정
        _openai_client = OpenAI()
    return _openai_client


# =========================
# 데이터 로딩
# =========================


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def _load_products_df() -> pd.DataFrame:
    global _products_df
    if _products_df is not None:
        return _products_df

    df = _load_csv(PRODUCTS_CSV)
    if df is None:
        # 비어 있으면라도 DataFrame 반환
        df = pd.DataFrame(columns=["name", "price"])

    # name/price 가 없으면 최대한 맞춰보기
    if "name" not in df.columns:
        # 첫 번째 문자열 컬럼을 name 으로 사용
        for c in df.columns:
            if df[c].dtype == object:
                df = df.rename(columns={c: "name"})
                break
    if "price" not in df.columns:
        df["price"] = None

    # 문자열화
    df["name"] = df["name"].astype(str)

    _products_df = df
    return _products_df


# =========================
# 상품 매칭 유틸
# =========================


def _normalize_item_name(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    name = raw.strip()
    if not name:
        return ""

    # 괄호 안 내용 제거: '까르보불닭볶음면(큰컵)' -> '까르보불닭볶음면'
    import re

    name = re.sub(r"\([^)]*\)", "", name)
    # ' 등', ' 외' 제거
    for suf in [" 등", " 외"]:
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name.strip()


def _split_items(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    tmp = (
        text.replace("+", ",")
        .replace("/", ",")
        .replace("·", ",")
        .replace("&", ",")
    )
    parts = [p.strip() for p in tmp.split(",")]
    return [p for p in parts if p]


def _match_product_to_official(
        raw_name: str, df_products: pd.DataFrame
) -> Tuple[str, Optional[int]]:
    if not isinstance(raw_name, str) or not raw_name.strip():
        return "", None

    base = _normalize_item_name(raw_name)
    if not base:
        return raw_name.strip(), None

    s = df_products["name"].astype(str)

    # 1) base 가 그대로 포함되는 상품
    contains = df_products[s.str.contains(base, na=False, regex=False)]

    # 2) 못 찾으면, 첫 단어/두 단어 기반 재시도
    if contains.empty:
        tokens = base.split()
        if len(tokens) >= 2:
            short = " ".join(tokens[:2])
        elif tokens:
            short = tokens[0]
        else:
            short = base

        contains = df_products[s.str.contains(short, na=False, regex=False)]

    if contains.empty:
        return raw_name.strip(), None

    # 여러 개면 이름이 가장 짧은 상품 선택
    row = contains.iloc[contains["name"].str.len().argmin()]
    official_name = str(row["name"])
    price = None
    if "price" in row and not pd.isna(row["price"]):
        try:
            price = int(row["price"])
        except Exception:
            price = None
    return official_name, price


# =========================
# 콤보 CSV -> 문서 생성
# =========================


def _build_combo_docs() -> List[Dict[str, Any]]:
    df_comb = _load_csv(COMBINATION_CSV)
    df_syn = _load_csv(SYNTHETIC_CSV)

    frames: List[pd.DataFrame] = []
    if df_comb is not None:
        frames.append(df_comb)
    if df_syn is not None:
        frames.append(df_syn)

    if not frames:
        print("[_build_combo_docs] combination/synthetic CSV 를 찾을 수 없습니다.")
        return []

    df_all = pd.concat(frames, ignore_index=True)

    # 필수 컬럼 존재 여부 확인
    required_cols = [
        "조합 이름",
        "주요 상품",
        "보조 상품(들)",
        "키워드 / 상황",
        "카테고리",
    ]
    for col in required_cols:
        if col not in df_all.columns:
            print(f"[_build_combo_docs] 필수 컬럼 누락: {col}")
            # 그래도 일단 진행 (해당 컬럼은 빈 문자열로 처리)

    df_products = _load_products_df()

    docs: List[Dict[str, Any]] = []

    for idx, row in df_all.iterrows():
        combo_name = str(row.get("조합 이름", "") or "").strip()
        main_product = str(row.get("주요 상품", "") or "").strip()
        sub_products = str(row.get("보조 상품(들)", "") or "").strip()
        situation = str(row.get("키워드 / 상황", "") or "").strip()
        category = str(row.get("카테고리", "") or "").strip() or "기타"

        # 이름/상품이 모두 비어 있으면 스킵
        if not combo_name and not main_product and not sub_products:
            continue

        raw_items: List[str] = []
        if main_product:
            # 주요 상품도 분리 로직 통일
            raw_items.extend(_split_items(main_product))
        if sub_products:
            raw_items.extend(_split_items(sub_products))

        # 그래도 없으면 원문 텍스트에서 최소 1개는 사용
        if not raw_items:
            if main_product:
                raw_items.append(main_product)
            elif sub_products:
                raw_items.append(sub_products)

        official_items: List[str] = []
        prices: List[Optional[int]] = []
        for raw_name in raw_items:
            official_name, price = _match_product_to_official(raw_name, df_products)
            if official_name:
                official_items.append(official_name)
                prices.append(price)

        if not official_items:
            # 최악의 경우 raw_items[0]만이라도 사용
            official_items = [raw_items[0]]
            prices = [None]

        total_price = sum(
            int(p) for p in prices if isinstance(p, (int, float))
        )
        total_price = int(total_price) if total_price > 0 else None

        # 임베딩용 텍스트
        items_text = ", ".join(official_items)
        semantic_parts = [combo_name, category, items_text]
        if situation:
            semantic_parts.append(f"상황: {situation}")
        semantic_text = " | ".join([p for p in semantic_parts if p])

        doc = {
            "id": idx,
            "name": combo_name or f"꿀조합 {idx + 1}",
            "category": category,
            "situation": situation,
            "items": official_items,       # CU 상품명
            "item_prices": prices,        # 각 상품 가격
            "total_price": total_price,   # 총합
            "semantic_text": semantic_text,
        }
        docs.append(doc)

    return docs


# =========================
# 임베딩 유틸
# =========================


def _embed_texts(texts: List[str]) -> np.ndarray:
    client = _get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")


def _embed_query(text: str) -> np.ndarray:
    client = _get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype="float32")
    return v


def _cosine_sim(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    return M_norm @ q_norm


# =========================
# precomputed 인덱스
# =========================


def build_precomputed_embeddings() -> None:
    docs = _build_combo_docs()
    print(f"[build_precomputed_embeddings] 총 {len(docs)} 개 조합 처리 중...")

    if not docs:
        print("[build_precomputed_embeddings] 생성된 콤보가 없습니다. CSV 컬럼을 다시 확인하세요.")
        return

    texts = [d["semantic_text"] for d in docs]
    embeds = _embed_texts(texts)

    with PRECOMP_DOCS_JSON.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    np.save(PRECOMP_EMB_NPY, embeds)

    print(f"[build_precomputed_embeddings] 저장 완료: {PRECOMP_DOCS_JSON}, {PRECOMP_EMB_NPY}")


def _load_semantic_index() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    global _combo_docs, _combo_embeddings

    if _combo_docs is not None and _combo_embeddings is not None:
        return _combo_docs, _combo_embeddings

    # 미리 계산된 파일 있으면 사용
    if PRECOMP_DOCS_JSON.exists() and PRECOMP_EMB_NPY.exists():
        with PRECOMP_DOCS_JSON.open("r", encoding="utf-8") as f:
            _combo_docs = json.load(f)
        _combo_embeddings = np.load(PRECOMP_EMB_NPY)
        return _combo_docs, _combo_embeddings

    # 없으면 즉석 생성
    docs = _build_combo_docs()
    if not docs:
        _combo_docs = []
        _combo_embeddings = np.zeros((0, 1536), dtype="float32")
        return _combo_docs, _combo_embeddings

    texts = [d["semantic_text"] for d in docs]
    embeds = _embed_texts(texts)

    _combo_docs = docs
    _combo_embeddings = embeds
    return _combo_docs, _combo_embeddings


# =========================
# 공개: 추천 함수
# =========================


def recommend_combos_openai_rag(user_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    docs, embeds = _load_semantic_index()
    if not docs or embeds.size == 0:
        return []

    q_vec = _embed_query(user_text)
    sims = _cosine_sim(q_vec, embeds)  # (N,)

    # 상위 top_k 인덱스
    top_idx = np.argsort(-sims)[:top_k]

    results: List[Dict[str, Any]] = []
    for rank, i in enumerate(top_idx):
        doc = docs[int(i)]
        score = float(sims[int(i)])

        items = doc.get("items", [])
        prices = doc.get("item_prices", [])
        total_price = doc.get("total_price")

        if total_price:
            price_text = f"{total_price:,}원"
        else:
            price_text = "가격 정보 없음"

        # 메인 추천(첫 번째)만 설명 붙이기
        if rank == 0:
            reason = (
                "입력하신 문장의 의미를 임베딩으로 분석해서 "
                "가장 잘 어울리는 편의점 꿀조합을 골랐어요. "
                f"(기준: '{user_text}')\n\n"
            )
            if doc.get("situation"):
                reason += f"이 조합은 '{doc['situation']}' 상황에 특히 잘 맞아요.\n\n"
            elif doc.get("category"):
                reason += f"이 조합은 '{doc['category']}' 카테고리 상황에 어울려요.\n\n"
            reason += f"이 조합을 모두 담으면 대략 {price_text} 정도예요."
        else:
            reason = ""

        results.append(
            {
                "name": doc.get("name", f"꿀조합 {i+1}"),
                "category": doc.get("category", "기타"),
                "items": items,
                "item_prices": prices,
                "total_price": total_price,
                "reason": reason,
                "similarity": score,
            }
        )

    return results


# =========================
# 공개: 카테고리 추론
# =========================


def infer_category_from_text(user_text: str) -> Optional[str]:
    docs, embeds = _load_semantic_index()
    if not docs or embeds.size == 0:
        return None

    q_vec = _embed_query(user_text)
    sims = _cosine_sim(q_vec, embeds)
    i = int(np.argmax(sims))
    cat = docs[i].get("category")
    return cat or None
