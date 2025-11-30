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
    """
    recommendation_service 가 반환하는 1개 결과(dict)를
    Kakao ItemCard 형태로 변환.
    """
    combo_name = result.get("name", "편의점 꿀조합")
    category = result.get("category", "")
    items = result.get("items", [])

    head = ItemCardHead(
        title=combo_name,
        description=f"카테고리: {category}" if category else None,
    )

    item_list: List[ListItem] = []
    for i, item in enumerate(items, start=1):
        # dict 형태(이름 + 가격)와 문자열 둘 다 지원
        if isinstance(item, dict):
            title_name = item.get("name") or "상품"
            price = item.get("price")
            desc = f"{price:,}원" if isinstance(price, (int, float)) else ""
        else:
            title_name = str(item)
            desc = ""
        item_list.append(
            ListItem(
                title=f"{i}. {title_name}",
                description=desc,
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
    """
    하단에 붙일 quickReplies 생성.
    user_text로부터 대략적인 category 를 추론해서 첫 번째 버튼에 반영.
    """
    inferred = infer_category_from_text(user_text) or "아무거나"

    base = [
        {
            "label": "다른 추천",
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

    base[0]["label"] = f"{inferred} 말고 다른 거"
    return base


@router.post(
    "/kakao/recommend",
    response_model=KakaoSkillResponse,
    response_model_exclude_none=True,
)
def kakao_recommend_combo(request: KakaoSkillRequest):
    """
    오픈빌더에서 들어온 사용자 발화를 기반으로
    RAG 추천 결과를 Kakao Skill JSON 으로 반환.
    """
    user_text = request.userRequest.utterance or ""
    if not user_text:
        user_text = "아무거나 추천해줘"

    # 1) 추천 로직 호출
    results = recommend_combos_openai_rag(user_text, top_k=3)

    # 추천이 하나도 없을 때: simpleText 로 안내
    if not results:
        simple = SimpleText(
            text="죄송해요, 지금은 추천할 꿀조합을 찾지 못했어요.\n다른 표현으로 다시 말해주실 수 있을까요?"
        )
        template = Template(
            outputs=[Component(simpleText=simple)],
            quickReplies=_build_quick_replies(user_text),
        )
        return KakaoSkillResponse(version="2.0", template=template)

    # 2) 첫 번째 추천을 ItemCard로, 나머지는 설명 텍스트에 포함
    main = results[0]
    item_card = _build_item_card_from_result(main)

    # 부가 설명 텍스트 (다른 후보들)
    other_lines = []
    for sub in results[1:]:
        other_lines.append(f"- {sub.get('name', '')} ({sub.get('category', '')})")

    desc_text = main.get("reason", "")
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
