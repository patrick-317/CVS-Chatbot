from typing import Any, Dict, List
from fastapi import APIRouter

from app.services.recommendation_service import (
    parse_user_preferences,
    recommend_combos_openai_rag,
    generate_combos_product2vec,
    UserPreferences,
)

router = APIRouter(prefix="/api/v1/kakao", tags=["recommendation"])


# ---------------------------------------------------------
# Quick Replies 생성
# ---------------------------------------------------------
def _build_quick_replies(_: str):
    return [
        {
            "label": "든든한 식사",
            "action": "message",
            "messageText": "든든한 식사 느낌으로 추천해줘"
        },
        {
            "label": "라면/분식",
            "action": "message",
            "messageText": "라면이나 분식류로 추천해줘"
        },
        {
            "label": "술안주/야식",
            "action": "message",
            "messageText": "술안주나 야식으로 추천해줘"
        },
        {
            "label": "간식/디저트",
            "action": "message",
            "messageText": "간식이나 디저트로 추천해줘"
        },
        {
            "label": "다이어트/건강",
            "action": "message",
            "messageText": "다이어트식이나 건강식으로 추천해줘"
        },
    ]


# ---------------------------------------------------------
# Kakao ItemCard 생성
# ---------------------------------------------------------
def _combo_to_itemcard_dict(combo) -> Dict[str, Any]:
    head = {
        "title": combo.name,
        "description": f"{combo.category} · 약 {combo.total_price or 0:,}원",
    }

    items: List[Dict[str, Any]] = []
    for i, it in enumerate(combo.items, start=1):
        price_txt = f"{it.price:,}원" if it.price else "가격 정보 없음"
        items.append(
            {
                "title": f"{i}. {it.name}",
                "description": price_txt,
            }
        )

    return {"head": head, "itemList": items}


# ---------------------------------------------------------
# Kakao SimpleText 텍스트 생성
# ---------------------------------------------------------
def _build_simple_text_str(
        user_text: str,
        main_combo,
        others: List[Any],
) -> str:
    lines: List[str] = []

    lines.append(f"입력하신 문장: \"{user_text}\"")
    lines.append("")
    lines.append("이 문장을 바탕으로 실제 꿀조합 데이터와")
    lines.append("CU 상품 패턴을 학습한 모델이 새로운 꿀조합을 만들어 봤어요.\n")

    lines.append(f"✅ 메인 추천: {main_combo.name}")
    lines.append(f"   · 카테고리: {main_combo.category}")
    if main_combo.total_price is not None:
        lines.append(f"   · 예상 가격: 약 {main_combo.total_price:,}원\n")

    if others:
        lines.append("📌 함께 어울리는 다른 꿀조합도 있어요:")
        for c in others:
            price_txt = f"{c.total_price:,}원" if c.total_price else "가격 정보 없음"
            lines.append(f"- {c.name} ({c.category}, 약 {price_txt})")

    return "\n".join(lines)


# ---------------------------------------------------------
# 조건에 맞는 추천이 없을 때 응답
# ---------------------------------------------------------
def _build_fail_response(user_text: str) -> Dict[str, Any]:
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": (
                            "요청하신 조건에 맞는 꿀조합을 찾지 못했어요.\n"
                            "조건을 조금 완화해서 다시 요청해 주세요.\n\n"
                            "예) '라면 제외하고 식사 느낌으로 추천해줘'"
                        )
                    }
                }
            ],
            "quickReplies": _build_quick_replies(user_text),
        },
    }


# ---------------------------------------------------------
# 메인 추천 엔드포인트
# ---------------------------------------------------------
@router.post("/recommend")
async def recommend(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    카카오 오픈빌더 → 편의점 꿀조합 추천 API
    사용자의 발화(utterance)를 기반으로
    1) 유저 선호 파싱
    2) CSV 기반 RAG 후보 탐색
    3) product2vec 생성형 후보 생성
    4) 최종 조합 구성
    """
    user_req = body.get("userRequest") or {}
    utterance = (user_req.get("utterance") or "").strip()
    user_text = utterance or "편의점 꿀조합 추천해줘"

    # (1) 입력 문장에서 선호/제약 파싱
    prefs: UserPreferences = parse_user_preferences(user_text)

    # (2) CSV 기반 RAG 후보
    rag_combos = recommend_combos_openai_rag(
        user_text=user_text,
        top_k=10,
        filters=prefs,
    )

    # (3) product2vec 기반 생성형 조합
    gen_combos = generate_combos_product2vec(
        user_text=user_text,
        base_candidates=rag_combos,
        max_new=3,
        filters=prefs,
    )

    # (4) 최종 후보 합치기
    all_combos = gen_combos + rag_combos
    if not all_combos:
        return _build_fail_response(user_text)

    main_combo = all_combos[0]
    others = all_combos[1:4]

    item_card_dict = _combo_to_itemcard_dict(main_combo)
    simple_text_str = _build_simple_text_str(user_text, main_combo, others)

    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {"itemCard": item_card_dict},
                {"simpleText": {"text": simple_text_str}},
            ],
            "quickReplies": _build_quick_replies(user_text),
        },
    }
