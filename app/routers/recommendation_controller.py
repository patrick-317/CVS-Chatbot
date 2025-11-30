from typing import List

from fastapi import APIRouter

from app.schemas.recommendation_model import (
    KakaoSkillRequest,
    KakaoSkillResponse,
    Template,
    Component,
    ItemCard,
    ListItem,
    ItemCardHead,
    SimpleText,
)
from app.services.recommendation_service import (
    recommend_combos_openai_rag,
    infer_category_from_text,
)

router = APIRouter(prefix="/api/v1", tags=["kakao"])


def _build_item_card_from_result(result: dict) -> ItemCard:
    combo_name = result.get("name", "편의점 꿀조합")
    category = result.get("category", "")
    items: List[str] = result.get("items", []) or []
    prices: List = result.get("item_prices", []) or []

    head = ItemCardHead(
        title=f"{combo_name}",
        description=f"카테고리: {category}" if category else None,
    )

    item_list: List[ListItem] = []
    for idx, name in enumerate(items, start=1):
        price = prices[idx - 1] if idx - 1 < len(prices) else None
        if isinstance(price, (int, float)):
            desc = f"{int(price):,}원"
        else:
            desc = "가격 정보 없음"

        item_list.append(
            ListItem(
                title=f"{idx}. {name}",
                description=desc,  # ★ 반드시 문자열 (빈 문자열도 허용)
                imageUrl=None,
            )
        )

    card = ItemCard(
        head=head,
        imageUrl=None,
        itemList=item_list,
    )
    return card


def _build_quick_replies(user_text: str) -> list[dict]:
    inferred = infer_category_from_text(user_text) or "아무거나"

    base = [
        {
            "label": f"{inferred} 말고 다른 거",
            "action": "message",
            "messageText": "다른 꿀조합 추천해줘",
        },
        {
            "label": "라면/분식",
            "action": "message",
            "messageText": "라면/분식 추천",
        },
        {
            "label": "식사류",
            "action": "message",
            "messageText": "식사류 추천",
        },
        {
            "label": "디저트",
            "action": "message",
            "messageText": "디저트 추천",
        },
    ]
    return base


@router.post(
    "/kakao/recommend",
    response_model=KakaoSkillResponse,
    response_model_exclude_none=True,
)
def kakao_recommend_combo(request: KakaoSkillRequest):
    user_text = request.userRequest.utterance or ""
    if not user_text:
        user_text = "아무거나 추천해줘"

    # 1) 추천 로직 호출 (OpenAI RAG)
    results = recommend_combos_openai_rag(user_text, top_k=3)

    # 2) 추천이 없으면 simpleText 로만 안내
    if not results:
        simple = SimpleText(
            text="죄송해요, 지금은 추천할 꿀조합을 찾지 못했어요.\n다른 표현으로 다시 말해주실 수 있을까요?"
        )
        template = Template(
            outputs=[Component(simpleText=simple)],
            quickReplies=_build_quick_replies(user_text),
        )
        return KakaoSkillResponse(version="2.0", template=template)

    # 3) 첫 번째 추천을 ItemCard 로, 나머지는 simpleText 하단에 후보로 표기
    main = results[0]
    item_card = _build_item_card_from_result(main)

    desc_text = main.get("reason", "")

    other_lines = []
    for sub in results[1:]:
        name = sub.get("name", "")
        cat = sub.get("category", "")
        if not name:
            continue
        if cat:
            other_lines.append(f"- {name} ({cat})")
        else:
            other_lines.append(f"- {name}")

    if other_lines:
        desc_text += "\n\n다른 추천 후보\n" + "\n".join(other_lines)

    simple = SimpleText(text=desc_text)

    template = Template(
        outputs=[
            Component(itemCard=item_card),
            Component(simpleText=simple),
        ],
        quickReplies=_build_quick_replies(user_text),
    )

    return KakaoSkillResponse(version="2.0", template=template)
