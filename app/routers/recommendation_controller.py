from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.recommendation_model import (
    KakaoSkillRequest,
    KakaoSkillResponse,
    Template,
    Component,
    ItemCard,
    ListItem,
    ItemCardHead,
    SimpleText,
    HoneyCombo,
)
from app.services.recommendation_service import (
    recommend_combos_openai_rag,
    parse_user_preferences,
    generate_combos_product2vec,
    UserPreferences,
)

router = APIRouter(prefix="/api/v1/kakao", tags=["recommendation"])

# ---------------------------------------------------------
# Quick Reply 빌더
# ---------------------------------------------------------


def _build_quick_replies(user_text: str):
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
# Kakao ItemCard / SimpleText 변환
# ---------------------------------------------------------


def _combo_to_itemcard(combo: HoneyCombo) -> ItemCard:
    head = ItemCardHead(
        title=combo.name,
        description=f"{combo.category} · 약 {combo.total_price or 0:,}원",
    )

    items: List[ListItem] = []
    for i, it in enumerate(combo.items, start=1):
        price_txt = f"{it.price:,}원" if it.price else "가격 정보 없음"
        items.append(
            ListItem(
                title=f"{i}. {it.name}",
                description=price_txt,
                imageUrl=None,
            )
        )

    return ItemCard(head=head, itemList=items)


def _build_simple_text(
        user_text: str,
        main_combo: HoneyCombo,
        others: List[HoneyCombo],
) -> SimpleText:
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

    return SimpleText(text="\n".join(lines))


# ---------------------------------------------------------
# 헬퍼: 실패 응답 공통 처리
# ---------------------------------------------------------


def _build_fail_response(user_text: str) -> KakaoSkillResponse:
    simple = SimpleText(
        text=(
            "요청하신 조건에 맞는 꿀조합을 찾지 못했어요.\n"
            "조건을 조금 완화해서 다시 요청해 주세요.\n\n"
            "예) '라면 제외하고 식사 느낌으로 추천해줘'"
        )
    )
    template = Template(
        outputs=[Component(simpleText=simple)],
        quickReplies=_build_quick_replies(user_text),
    )
    return KakaoSkillResponse(version="2.0", template=template)


# ---------------------------------------------------------
# 메인 엔드포인트
# ---------------------------------------------------------


@router.post("/recommend", response_model=KakaoSkillResponse)
def recommend(request: KakaoSkillRequest) -> KakaoSkillResponse:
    """
    카카오 오픈빌더용 편의점 꿀조합 추천 엔드포인트.
    POST /api/v1/kakao/recommend
    """
    if request.userRequest is None:
        raise HTTPException(status_code=400, detail="userRequest가 비어 있습니다.")

    user_text = (request.userRequest.utterance or "").strip()
    if not user_text:
        return _build_fail_response(user_text)

    # 1) 유저 선호 파싱 (라면 제외, 다이어트, 카테고리, 술 여부 등)
    prefs: UserPreferences = parse_user_preferences(user_text)

    # 2) combo CSV 기반 꿀조합 후보 (RAG-lite 역할)
    rag_combos: List[HoneyCombo] = recommend_combos_openai_rag(
        user_text=user_text,
        top_k=10,
        filters=prefs,
    )

    # 3) product2vec 기반 새로운 조합 생성 시도 (현재는 비어 있거나, 추후 확장용)
    gen_combos: List[HoneyCombo] = generate_combos_product2vec(
        user_text=user_text,
        base_candidates=rag_combos,
        max_new=3,
        filters=prefs,
    )

    # 4) 결과 합치기 (생성형 + 데이터셋 기반)
    all_combos: List[HoneyCombo] = gen_combos + rag_combos

    if not all_combos:
        return _build_fail_response(user_text)

    # 메인 추천 + 서브 추천
    main_combo = all_combos[0]
    others = all_combos[1:4]

    item_card = _combo_to_itemcard(main_combo)
    simple = _build_simple_text(user_text, main_combo, others)

    template = Template(
        outputs=[
            Component(itemCard=item_card),
            Component(simpleText=simple),
        ],
        quickReplies=_build_quick_replies(user_text),
    )

    return KakaoSkillResponse(version="2.0", template=template)
