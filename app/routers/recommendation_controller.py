from typing import Any, Dict, List
from fastapi import APIRouter

# [수정 1] 비동기 함수로 Import 변경 (generate_combos_product2vec은 속도 문제로 제외하거나 나중에 async로 구현 필요)
from app.services.recommendation_service import (
    parse_user_preferences,
    recommend_combos_openai_rag_async,
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
    # 콤보 객체의 속성에 안전하게 접근 (None 체크 등)
    total_price = combo.total_price if combo.total_price else 0
    head = {
        "title": combo.name,
        "description": f"{combo.category} · 약 {total_price:,}원",
    }

    items: List[Dict[str, Any]] = []
    # 아이템 최대 5개까지만 노출 (카카오 제한 고려)
    for i, it in enumerate(combo.items[:5], start=1):
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

    # lines.append(f"입력하신 문장: \"{user_text}\"") # (선택) 길이 줄이기를 위해 주석 처리 가능
    lines.append("요청하신 느낌에 딱 맞는 편의점 꿀조합을 찾아왔어요! 🏪✨")
    lines.append("")

    lines.append(f"✅ [메인 추천] {main_combo.name}")
    lines.append(f"   · 종류: {main_combo.category}")
    if main_combo.total_price is not None:
        lines.append(f"   · 예상 가격: 약 {main_combo.total_price:,}원\n")

    if others:
        lines.append("👇 다른 추천 조합도 구경해보세요:")
        for c in others:
            price_txt = f"{c.total_price:,}원" if c.total_price else "-"
            lines.append(f"• {c.name} ({c.category}, {price_txt})")

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
                            "죄송해요, 요청하신 조건에 딱 맞는 꿀조합을 찾지 못했어요. 😢\n"
                            "조건을 조금 더 단순하게 말씀해 주시겠어요?\n\n"
                            "예) '매운 라면 조합 추천해줘', '5000원 이하 식사'"
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
    (비동기 최적화 적용 버전)
    """
    user_req = body.get("userRequest") or {}
    utterance = (user_req.get("utterance") or "").strip()
    user_text = utterance or "편의점 꿀조합 추천해줘"

    # (1) 입력 문장에서 선호/제약 파싱 (CPU 작업이므로 동기 실행)
    prefs: UserPreferences = parse_user_preferences(user_text)

    # (2) CSV 기반 RAG 후보 탐색 (비동기 I/O 적용)
    # [수정 2] await 키워드 추가 및 async 함수명 사용
    # [수정 3] 타임아웃 방지를 위해 생성형(Product2Vec) 로직은 제외하고 RAG 결과만 활용
    all_combos = await recommend_combos_openai_rag_async(
        user_text=user_text,
        top_k=5,  # 속도를 위해 개수 조정
        filters=prefs,
    )

    if not all_combos:
        return _build_fail_response(user_text)

    main_combo = all_combos[0]
    others = all_combos[1:4] # 메인 제외 최대 3개

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