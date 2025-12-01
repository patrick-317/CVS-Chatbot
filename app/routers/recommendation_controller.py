from typing import List, Dict, Any

from fastapi import APIRouter

from app.services.recommendation_service import (
    recommend_combos_openai_rag,
    parse_user_preferences,
    generate_combos_product2vec,
    UserPreferences,
)

# 카카오 오픈빌더에서 설정한 스킬 URL: /api/v1/kakao/recommend
router = APIRouter(prefix="/api/v1/kakao", tags=["recommendation"])


# ---------------------------------------------------------
# Quick Reply 빌더
# ---------------------------------------------------------


def _build_quick_replies(user_text: str) -> List[Dict[str, Any]]:
    return [
        {
            "label": "다시 추천받기",
            "action": "message",
            "messageText": user_text or "편의점 꿀조합 추천해줘",
        },
        {
            "label": "식사류 추천",
            "action": "message",
            "messageText": "식사 느낌으로 꿀조합 추천해줘",
        },
        {
            "label": "라면 제외",
            "action": "message",
            "messageText": "라면 제외하고 추천해줘",
        },
    ]


# ---------------------------------------------------------
# Kakao ItemCard / SimpleText 변환 (딕셔너리 기반)
# ---------------------------------------------------------


def _combo_to_itemcard(combo) -> Dict[str, Any]:
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
                "imageUrl": None,
            }
        )

    return {
        "head": head,
        "itemList": items,
    }


def _build_simple_text(
        user_text: str,
        main_combo,
        others: List[Any],
) -> str:
    lines: List[str] = []

    lines.append(f"입력하신 문장: \"{user_text}\"")
    lines.append("")
    lines.append("이 문장을 바탕으로 실제 꿀조합 데이터와")
    lines.append("CU 상품 패턴을 학습한 모델이 새로운 꿀조합을 만들어 봤어요.")
    lines.append("")
    lines.append(f"✅ 메인 추천: {main_combo.name}")
    lines.append(f"   · 카테고리: {main_combo.category}")
    if main_combo.total_price is not None:
        lines.append(f"   · 예상 가격: 약 {main_combo.total_price:,}원")
    lines.append("")

    if others:
        lines.append("📌 함께 어울리는 다른 꿀조합도 있어요:")
        for c in others:
            price_txt = f"{c.total_price:,}원" if c.total_price else "가격 정보 없음"
            lines.append(f"- {c.name} ({c.category}, 약 {price_txt})")

    return "\n".join(lines)


# ---------------------------------------------------------
# 실패 응답
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
# 메인 엔드포인트 (스키마 없이 dict로 처리)
# ---------------------------------------------------------


@router.post("/recommend")
async def recommend(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    카카오 오픈빌더용 편의점 꿀조합 추천 엔드포인트.
    POST /api/v1/kakao/recommend
    """
    user_req = (body.get("userRequest") or {})
    utterance = (user_req.get("utterance") or "").strip()

    user_text = utterance or "편의점 꿀조합 추천해줘"

    # 1) 유저 선호 파싱
    prefs: UserPreferences = parse_user_preferences(user_text)

    # 2) combo CSV 기반 후보
    rag_combos = recommend_combos_openai_rag(
        user_text=user_text,
        top_k=10,
        filters=prefs,
    )

    # 3) product2vec 기반 생성 (현재는 비워둔 상태일 수 있음)
    gen_combos = generate_combos_product2vec(
        user_text=user_text,
        base_candidates=rag_combos,
        max_new=3,
        filters=prefs,
    )

    all_combos = gen_combos + rag_combos

    if not all_combos:
        return _build_fail_response(user_text)

    main_combo = all_combos[0]
    others = all_combos[1:4]

    item_card_dict = _combo_to_itemcard(main_combo)
    simple_text_str = _build_simple_text(user_text, main_combo, others)

    response: Dict[str, Any] = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "itemCard": item_card_dict,
                    "simpleText": None,
                },
                {
                    "simpleText": {"text": simple_text_str},
                    "itemCard": None,
                },
            ],
            "quickReplies": _build_quick_replies(user_text),
        },
    }

    return response
