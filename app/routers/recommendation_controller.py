from typing import Optional, Dict, Any, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.schemas.recommendation_model import (
    KakaoSkillResponse,
    Template,
    Component,
    ItemCard,
    ListItem,
    ItemCardHead,
    SimpleText,
)
from app.services.recommendation_service import recommend_combos_openai_rag


# ---------- 카카오에서 오는 요청 바디 스키마 ----------


class KakaoUserRequest(BaseModel):
    timezone: Optional[str] = None
    utterance: str
    params: Dict[str, Any] = Field(default_factory=dict)
    block: Optional[Dict[str, Any]] = None
    user: Optional[Dict[str, Any]] = None


class KakaoAction(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    detailParams: Dict[str, Any] = Field(default_factory=dict)


class KakaoSkillPayload(BaseModel):
    userRequest: KakaoUserRequest
    action: KakaoAction
    bot: Optional[Dict[str, Any]] = None
    intent: Optional[Dict[str, Any]] = None


router = APIRouter(prefix="/api/v1", tags=["kakao"])


# ---------- 응답용 헬퍼 ----------


def _build_item_card_from_result(result: Dict[str, Any]) -> ItemCard:
    """
    recommendation_service.recommend_combos_openai_rag() 의 1개 결과를
    Kakao itemCard 로 변환
    """

    # 조합 이름
    combo_name = (
            result.get("ai_combo_name")
            or result.get("name")
            or "편의점 꿀조합"
    )

    # 카테고리
    category = result.get("category") or "기타"

    head = ItemCardHead(
        title=combo_name,
        description=f"카테고리: {category}",
    )

    # 아이템 리스트 (실제 구매해야 하는 상품들)
    items: List[Dict[str, Any]] = result.get("items", [])
    item_list: List[ListItem] = []

    for idx, item in enumerate(items, start=1):
        # 서비스에서 내려주는 키 이름 여러 경우 방어
        name = (
                item.get("name")
                or item.get("product_name")
                or item.get("item_name")
                or "상품"
        )

        price = item.get("price") or item.get("unit_price")
        if isinstance(price, (int, float)):
            price_text = f"{price:,}원"
        elif isinstance(price, str) and price.strip():
            price_text = price
        else:
            price_text = ""

        item_list.append(
            ListItem(
                title=f"{idx}. {name}",
                description=price_text,
                imageUrl=None,
            )
        )

    return ItemCard(
        head=head,
        imageUrl=None,
        itemList=item_list,
    )


def _build_quick_replies() -> List[Dict[str, Any]]:
    """하단 고정 버튼 (간단 버전)"""
    return [
        {
            "label": "아무거나 말고 다른 거",
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


# ---------- 메인 엔드포인트 ----------


@router.post(
    "/kakao/recommend",
    response_model=KakaoSkillResponse,
    response_model_exclude_none=True,
)
def kakao_recommend_combo(payload: KakaoSkillPayload):
    """
    카카오 오픈빌더 스킬 엔드포인트(A안).
    - userRequest.utterance 만 받아서 서버에서 전부 추천 생성
    """

    user_text = (payload.userRequest.utterance or "").strip()
    if not user_text:
        user_text = "아무거나 추천해줘"

    # 1) 추천 로직 실행 (RAG + 딥러닝 포함)
    results = recommend_combos_openai_rag(user_text, top_k=3)

    # 추천이 없을 때: 안내 메시지
    if not results:
        simple = SimpleText(
            text="죄송해요, 지금은 추천할 꿀조합을 찾지 못했어요.\n다른 표현으로 다시 말해주실 수 있을까요?"
        )
        template = Template(
            outputs=[Component(simpleText=simple)],
            quickReplies=_build_quick_replies(),
        )
        return KakaoSkillResponse(version="2.0", template=template)

    # 2) 대표 추천 1개를 카드로, 나머지는 텍스트에 요약
    main = results[0]
    item_card = _build_item_card_from_result(main)

    # 메인 설명/이유 + 총 가격
    reason = main.get("reason") or ""
    total_price = main.get("total_price")
    if isinstance(total_price, (int, float)):
        reason += f"\n\n이 조합을 모두 담으면 대략 {total_price:,}원 정도예요."

    # 다른 후보들
    other_lines: List[str] = []
    for sub in results[1:]:
        name = sub.get("ai_combo_name") or sub.get("name") or ""
        cat = sub.get("category") or ""
        if name:
            if cat:
                other_lines.append(f"- {name} ({cat})")
            else:
                other_lines.append(f"- {name}")

    if other_lines:
        if not reason.endswith("\n\n") and reason:
            reason += "\n\n"
        reason += "다른 추천 후보\n" + "\n".join(other_lines)

    simple = SimpleText(
        text=(
                f"입력하신 문장의 의미를 임베딩으로 분석해서 "
                f"가장 비슷한 분위기의 꿀조합을 골랐어요. (기준: '{user_text}')\n\n"
                + reason
        )
    )

    template = Template(
        outputs=[
            Component(itemCard=item_card),
            Component(simpleText=simple),
        ],
        quickReplies=_build_quick_replies(),
    )

    return KakaoSkillResponse(version="2.0", template=template)
