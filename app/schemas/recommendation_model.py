from typing import List, Dict, Any, Optional
from pydantic import BaseModel


# ---------------------------------------------------------
# Kakao Request / Response 스키마
# ---------------------------------------------------------


class KakaoUser(BaseModel):
    id: str
    type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None


class KakaoUserRequest(BaseModel):
    timezone: str
    params: Dict[str, Any] = {}
    block: Dict[str, Any]
    utterance: str
    lang: Optional[str] = None
    user: KakaoUser


class KakaoAction(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    params: Dict[str, Any] = {}
    detailParams: Dict[str, Any] = {}


class KakaoBot(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None


class KakaoSkillRequest(BaseModel):
    intent: Optional[Dict[str, Any]] = None
    userRequest: KakaoUserRequest
    bot: KakaoBot
    action: KakaoAction


# ---------------------------------------------------------
# Kakao Response 스키마
# ---------------------------------------------------------


class SimpleText(BaseModel):
    text: str


class ItemCardHead(BaseModel):
    title: str
    description: Optional[str] = None


class ListItem(BaseModel):
    title: str
    description: Optional[str] = None
    imageUrl: Optional[str] = None


class ItemCard(BaseModel):
    head: ItemCardHead
    itemList: List[ListItem]


class Component(BaseModel):
    simpleText: Optional[SimpleText] = None
    itemCard: Optional[ItemCard] = None


class Template(BaseModel):
    outputs: List[Component]
    quickReplies: Optional[List[Dict[str, Any]]] = None


class KakaoSkillResponse(BaseModel):
    version: str = "2.0"
    template: Template


# ---------------------------------------------------------
# 내부에서 사용하는 콤보 / 아이템 구조
# ---------------------------------------------------------


class ComboItem(BaseModel):
    original_name: str
    name: str
    price: Optional[int] = None
    main_category: Optional[str] = None  # CU main_category (간편식사, 과자류, 생활용품 등)


class HoneyCombo(BaseModel):
    id: int
    name: str
    category: str
    items: List[ComboItem]
    total_price: Optional[int] = None
    mood: Optional[str] = None
    generated: bool = False
