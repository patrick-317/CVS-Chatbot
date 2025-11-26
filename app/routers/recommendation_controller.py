from fastapi import APIRouter

from app.schemas.recommendation_model import KakaoSkillRequest, KakaoSkillResponse, Template, Component, ItemCard, ListItem, ItemCardHead



router = APIRouter(prefix="/api/v1", tags=["kakao"])



@router.post("/kakao/recommend", response_model=KakaoSkillResponse, response_model_exclude_none=True)

def kakao_recommend_combo(request: KakaoSkillRequest):

    keyword = request.action.params.get("keyword", "")

    category = request.action.params.get("category", "")



    item_card = ItemCard(

        head=ItemCardHead(

            title="악마의 유혹 디저트",

            description="스트레스 폭발할 때 딱 좋은 달콤한 조합"

        ),

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
