from fastapi import APIRouter
from app.schemas.recommendation_model import KakaoSkillRequest, KakaoSkillResponse, Template, Component, ItemCard, ListItem

router = APIRouter(prefix="/api/v1", tags=["kakao"])


@router.post("/kakao/recommend", response_model=KakaoSkillResponse)
def kakao_recommend_combo(request: KakaoSkillRequest):
    keyword = request.userRequest.action.params.get("keyword", "")
    category = request.userRequest.action.params.get("category", "")

    # TODO: 내부 로직 구현 후 더미데이터 응답 로직 수정 필요    
    item_card = ItemCard(
        title="악마의 유혹 디저트",
        description="스트레스 폭발할 때 딱 좋은 달콤한 조합",
        itemList=[
            ListItem(
                title="주요 상품",
                description="브라우니"
            ),
            ListItem(
                title="보조 상품",
                description="악마빙수, 아메리카노"
            )
        ]
    )
    
    template = Template(
        outputs=[
            Component(itemCard=item_card)
        ]
    )
    
    return KakaoSkillResponse(
        version="2.0",
        template=template
    )
