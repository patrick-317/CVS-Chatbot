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


# ---------- Kakao 응답 구성 유틸 ----------

def _build_item_card_from_result(result: dict) -> ItemCard:
    """
    recommendation_service 결과(dict)를 Kakao ItemCard 로 변환.

    result 예시:
    {
        "name": "다이어트 간식",
        "category": "건강/다이어트",
        "items": [
            {"name": "샐)오리지널닭가슴살샐러", "price": 4800},
            {"name": "샐)허니리코타치즈샐러드", "price": 4800},
        ],
        "total_price": 9600,
        "mood": "다이어트, 건강, 운동, 식단 관리"
    }
    """
    combo_name = result.get("name", "편의점 꿀조합")
    category = result.get("category", "")
    items = result.get("items", [])

    head = ItemCardHead(
        title=f"{combo_name}",
        description=f"카테고리: {category}" if category else None,
    )

    item_list: list[ListItem] = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, dict):
            item_name = item.get("name") or item.get("original_name") or ""
            price = item.get("price")
        else:
            item_name = str(item)
            price = None

        if isinstance(price, (int, float)):
            desc = f"{int(price):,}원"
        else:
            desc = ""

        item_list.append(
            ListItem(
                title=f"{i}. {item_name}",
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


# ---------- 엔드포인트 ----------

@router.post(
    "/kakao/recommend",
    response_model=KakaoSkillResponse,
    response_model_exclude_none=True,
)
def kakao_recommend_combo(request: KakaoSkillRequest):
    """
    카카오 오픈빌더에서 호출하는 메인 엔드포인트.
    - userRequest.utterance 하나만 받아서, 바로 꿀조합 추천.
    """
    user_text = request.userRequest.utterance or ""
    if not user_text.strip():
        user_text = "아무거나 추천해줘"

    # 1) 추천 로직 (RAG)
    results = recommend_combos_openai_rag(user_text, top_k=3)

    # 2) 추천 없음 처리
    if not results:
        simple = SimpleText(
            text="죄송해요, 지금은 추천할 꿀조합을 찾지 못했어요.\n다른 표현으로 다시 말해주실 수 있을까요?"
        )
        template = Template(
            outputs=[Component(simpleText=simple)],
            quickReplies=_build_quick_replies(user_text),
        )
        return KakaoSkillResponse(version="2.0", template=template)

    # 3) 메인 카드 + 부가 설명
    main = results[0]
    item_card = _build_item_card_from_result(main)

    desc_lines: list[str] = []

    desc_lines.append(
        f"입력하신 문장의 의미를 임베딩으로 분석해서 가장 비슷한 분위기의 꿀조합을 골랐어요. (기준: '{user_text}')"
    )

    mood = main.get("mood")
    if mood:
        desc_lines.append(f"이 조합은 '{mood}' 상황에 잘 어울려요.")

    total_price = main.get("total_price")
    if isinstance(total_price, (int, float)):
        desc_lines.append(
            f"이 조합을 모두 담으면 대략 {int(total_price):,}원 정도예요."
        )
    else:
        desc_lines.append("이 조합을 모두 담으면 대략 가격 정보 없음 정도예요.")

    # 다른 추천 후보
    if len(results) > 1:
        other_lines = []
        for sub in results[1:]:
            other_lines.append(
                f"- {sub.get('name', '')} ({sub.get('category', '')})"
            )
        if other_lines:
            desc_lines.append("\n다른 추천 후보")
            desc_lines.extend(other_lines)

    simple = SimpleText(text="\n\n".join(desc_lines))

    template = Template(
        outputs=[
            Component(itemCard=item_card),
            Component(simpleText=simple),
        ],
        quickReplies=_build_quick_replies(user_text),
    )

    return KakaoSkillResponse(version="2.0", template=template)
