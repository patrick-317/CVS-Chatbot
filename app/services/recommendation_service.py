import os
import re
import json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

# =====================================================================
# 경로 설정
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
COMBINATION_PATH = os.path.join(BASE_DIR, "combination.csv")
SYNTHETIC_PATH = os.path.join(BASE_DIR, "synthetic_honey_combos_1000.csv")
CU_PRODUCTS_PATH = os.path.join(BASE_DIR, "cu_official_products.csv")

PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed")
COMBO_EMBEDDINGS_PATH = os.path.join(PRECOMPUTED_DIR, "combo_embeddings.npy")
COMBO_DOCS_PATH = os.path.join(PRECOMPUTED_DIR, "combo_docs.json")

# =====================================================================
# 전역 캐시
# =====================================================================
_comb_df: Optional[pd.DataFrame] = None
_syn_df: Optional[pd.DataFrame] = None
_cu_df: Optional[pd.DataFrame] = None
_combo_df: Optional[pd.DataFrame] = None

_keyword_dict: Optional[Dict[str, set]] = None

_openai_client: Optional[OpenAI] = None
_openai_embedding_model: str = "text-embedding-3-small"

_combo_embeddings: Optional[np.ndarray] = None  # N x d

# =====================================================================
# 🔹 카테고리 추론용 키워드 (컨트롤러에서 사용)
# =====================================================================
CATEGORY_KEYWORDS = {
    "라면/분식": ["라면", "컵라면", "국물라면", "떡볶이", "분식", "우동", "튀김", "어묵"],
    "식사류": ["밥", "식사", "도시락", "덮밥", "카레", "죽", "파스타", "볶음밥"],
    "간편식": ["삼각김밥", "주먹밥", "햄버거", "샌드위치", "핫도그", "토스트"],
    "디저트": ["빵", "케이크", "쿠키", "초코", "젤리", "아이스크림", "빙수", "달달", "달콤"],
    "술안주/야식": ["맥주", "소주", "와인", "안주", "야식", "치킨", "포차", "편맥"],
}


def infer_category_from_text(text: str) -> str:
    """사용자 문장에서 대략적인 카테고리 추론 (quickReplies용)"""
    if not text:
        return ""

    low = text.lower()
    best_cat = ""
    best_score = 0

    for cat, kws in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in text or kw in low:
                score += 1
        if score > best_score:
            best_cat = cat
            best_score = score

    return best_cat


# =====================================================================
# OpenAI client
# =====================================================================
def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


# =====================================================================
# 텍스트 전처리
# =====================================================================
def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\(.*?\)", "", text)
    t = re.sub(r"[^0-9a-zA-Z가-힣]", "", t)
    return t.lower()


# =====================================================================
# 데이터 로딩
# =====================================================================
def _load_data():
    global _comb_df, _syn_df, _cu_df, _combo_df, _keyword_dict

    if _combo_df is not None:
        return

    _comb_df = pd.read_csv(COMBINATION_PATH)
    _syn_df = pd.read_csv(SYNTHETIC_PATH)
    _cu_df = pd.read_csv(CU_PRODUCTS_PATH)

    _comb_df["source"] = "real"
    _syn_df["source"] = "synthetic"

    _combo_df = pd.concat([_comb_df, _syn_df], ignore_index=True)

    for col in ["조합 이름", "주요 상품", "보조 상품(들)", "키워드 / 상황", "카테고리", "source"]:
        if col in _combo_df.columns:
            _combo_df[col] = _combo_df[col].fillna("")

    if "clean_name" not in _cu_df.columns:
        _cu_df["clean_name"] = _cu_df["name"].apply(_clean_text)

    _keyword_dict = _build_keyword_dict()


def _build_keyword_dict() -> Dict[str, set]:
    global _combo_df
    d: Dict[str, set] = {}
    for val in _combo_df["키워드 / 상황"]:
        if not isinstance(val, str):
            continue

        parts = re.split(r"[;,]", val)
        for p in parts:
            p = p.strip()
            if not p:
                continue

            low = p.lower()
            compact = low.replace(" ", "")
            for k in {low, compact}:
                d.setdefault(k, set()).add(p)
    return d


# =====================================================================
# RAG 기반: 각 조합이 왜 좋은지 한 줄 설명 생성 (오프라인용)
# =====================================================================
def _rag_extract_combo_features(row: pd.Series) -> str:
    prompt = f"""
    아래 편의점 꿀조합이 왜 좋은 조합인지 설명해줘.
    '맛 조화(매운/단/짠/고소)', '식감 대비(바삭/쫀득/부드러움)', 
    '온도 대비(뜨거움+차가움)', '중화/밸런스(매운맛+치즈, 짠맛+단맛)', 
    '포만감', '상황(야식/다이어트/간편식)' 관점에서 1~2문장으로 요약해줘.

    주요상품: {row['주요 상품']}
    보조상품: {row['보조 상품(들)']}
    상황/키워드: {row['키워드 / 상황']}
    """
    resp = _get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# =====================================================================
# B안 핵심: 임베딩을 미리 계산해서 파일로 저장 (한 번만 실행)
# =====================================================================
def build_precomputed_embeddings():
    _load_data()

    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

    docs: List[str] = []
    print(f"[build_precomputed_embeddings] 총 {_combo_df.shape[0]} 개 조합 처리 중...")

    for _, row in _combo_df.iterrows():
        try:
            reason = _rag_extract_combo_features(row)
        except Exception:
            # 실패 시 이유 없이도 진행
            reason = ""

        doc = " / ".join(
            [
                f"조합 이름: {row['조합 이름']}",
                f"주요 상품: {row['주요 상품']}",
                f"보조 상품: {row['보조 상품(들)']}",
                f"상황: {row['키워드 / 상황']}",
                f"카테고리: {row['카테고리']}",
                f"이유: {reason}",
            ]
        )
        docs.append(doc)

    # 임베딩 계산
    client = _get_openai_client()
    embeddings: List[List[float]] = []
    batch_size = 100

    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        resp = client.embeddings.create(
            model=_openai_embedding_model,
            input=chunk,
        )
        for d in resp.data:
            embeddings.append(d.embedding)

    arr = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    arr = arr / norms

    np.save(COMBO_EMBEDDINGS_PATH, arr)

    with open(COMBO_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"[build_precomputed_embeddings] 저장 완료: {COMBO_EMBEDDINGS_PATH}")


# =====================================================================
# 서버에서 사용하는 임베딩 로더 (빠른 경로)
# =====================================================================
def _load_semantic_index():
    global _combo_embeddings
    _load_data()

    if _combo_embeddings is not None:
        return

    if os.path.exists(COMBO_EMBEDDINGS_PATH):
        arr = np.load(COMBO_EMBEDDINGS_PATH)
        _combo_embeddings = arr.astype("float32")
        return

    # 🔻 fallback: RAG 이유 없이 간단 텍스트로 임베딩 생성 (최초 1회)
    client = _get_openai_client()
    docs: List[str] = []
    for _, row in _combo_df.iterrows():
        doc = " / ".join(
            [
                f"조합 이름: {row['조합 이름']}",
                f"주요 상품: {row['주요 상품']}",
                f"보조 상품: {row['보조 상품(들)']}",
                f"상황: {row['키워드 / 상황']}",
                f"카테고리: {row['카테고리']}",
            ]
        )
        docs.append(doc)

    embeddings: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        resp = client.embeddings.create(
            model=_openai_embedding_model,
            input=chunk,
        )
        for d in resp.data:
            embeddings.append(d.embedding)

    arr = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    arr = arr / norms

    _combo_embeddings = arr
    # 원하면 여기서도 npy로 저장 가능
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    np.save(COMBO_EMBEDDINGS_PATH, arr)


# =====================================================================
# 키워드 추출
# =====================================================================
def extract_keywords(text: str) -> List[str]:
    raw = text.lower()
    compact = re.sub(r"[^0-9a-zA-Z가-힣]", "", raw)

    found = set()
    for trig, concept_set in _keyword_dict.items():
        if trig in compact:
            found |= concept_set

    if not found:
        parts = re.split(r"\s+|[,./!?]", raw)
        found = {p for p in parts if len(p) >= 2}

    return list(found)


# =====================================================================
# CU 상품 매칭 (비식품 필터)
# =====================================================================
def _is_food_product(row: pd.Series) -> bool:
    name = str(row.get("name", "")).lower()
    non_food = ["우산", "이어폰", "충전", "usb", "라이터", "물티슈", "건전지"]
    for n in non_food:
        if n in name:
            return False
    return True


def _find_cu_products(row: pd.Series, max_items: int = 3) -> List[str]:
    global _cu_df
    _load_data()

    combo_items = f"{row['주요 상품']},{row['보조 상품(들)']}"
    parts = re.split(r"[,+/·]|외", combo_items)

    results: List[str] = []

    for item in parts:
        item = item.strip()
        if not item:
            continue

        clean = _clean_text(item)
        best = None
        best_score = 0

        for _, cu in _cu_df.iterrows():
            cu_clean = cu["clean_name"]
            score = 0
            if clean and clean in cu_clean:
                score = len(clean)
            elif cu_clean and cu_clean in clean:
                score = len(cu_clean)

            if score > best_score:
                best_score = score
                best = cu["name"]

        if not best:
            continue

        cu_row = _cu_df[_cu_df["name"] == best]
        if cu_row.empty or not _is_food_product(cu_row.iloc[0]):
            continue

        if best not in results:
            results.append(best)

        if len(results) >= max_items:
            break

    return results


# =====================================================================
# 🔥 최종 추천 함수 — 컨트롤러에서 호출
# =====================================================================
def recommend_combos_openai_rag(user_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    global _combo_embeddings, _combo_df

    _load_data()
    _load_semantic_index()

    if not user_text:
        user_text = "아무거나 추천해줘"

    client = _get_openai_client()

    # 1) 사용자 문장 임베딩 (한 번)
    resp = client.embeddings.create(
        model=_openai_embedding_model,
        input=[user_text],
    )
    q = np.array(resp.data[0].embedding, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-10)

    # 2) 코사인 유사도
    sims = _combo_embeddings @ q

    # 3) 키워드 기반 스코어
    keywords = extract_keywords(user_text)
    kw_scores = []
    for _, row in _combo_df.iterrows():
        txt = " ".join(
            [
                row["조합 이름"],
                row["주요 상품"],
                row["보조 상품(들)"],
                row["키워드 / 상황"],
                row["카테고리"],
            ]
        ).lower()
        score = sum(1 for kw in keywords if kw.lower() in txt)
        kw_scores.append(score)

    kw_scores = np.array(kw_scores, dtype="float32")
    if kw_scores.max() > 0:
        kw_scores /= (kw_scores.max() + 1e-10)

    # 4) 실제 꿀조합(real) 보너스
    is_real = (_combo_df["source"] == "real").astype("float32").to_numpy()

    # 5) 최종 점수
    final = 0.75 * sims + 0.20 * kw_scores + 0.05 * is_real

    ordered = list(np.argsort(-final))

    kw_preview = ", ".join(keywords[:3]) if keywords else user_text[:20]

    results: List[Dict[str, Any]] = []

    for idx in ordered:
        row = _combo_df.iloc[idx]
        items = _find_cu_products(row, max_items=3)
        if len(items) < 2:
            continue

        reason = (
            "입력하신 문장의 의미를 임베딩으로 분석해서 "
            f"가장 비슷한 분위기의 꿀조합을 골랐어요. (기준: '{kw_preview}')"
        )
        if row["source"] == "real":
            reason += "\n실제로 많이 알려진 꿀조합이라서 우선적으로 추천했어요."

        results.append(
            {
                "name": row["조합 이름"],
                "category": row["카테고리"],
                "reason": reason,
                "items": items,
            }
        )

        if len(results) >= top_k:
            break

    return results
