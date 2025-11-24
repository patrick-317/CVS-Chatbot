from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class KakaoUserRequest(BaseModel):
    timezone: str
    id: str
    properties: Optional[Dict[str, Any]] = None


class KakaoBlock(BaseModel):
    id: str
    name: str


class KakaoIntent(BaseModel):
    id: str
    name: str


class KakaoAction(BaseModel):
    id: str
    name: str
    params: Dict[str, str]
    detailParams: Optional[Dict[str, Any]] = None
    clientExtra: Optional[Dict[str, Any]] = None


class KakaoRequest(BaseModel):
    user: KakaoUserRequest
    utterance: str
    lang: Optional[str] = None
    timezone: str
    block: KakaoBlock
    intent: KakaoIntent
    action: KakaoAction


class KakaoSkillRequest(BaseModel):
    version: str = "2.0"
    userRequest: KakaoRequest
    contexts: Optional[List[Dict[str, Any]]] = []


class SimpleText(BaseModel):
    text: str


class ListItem(BaseModel):
    title: str
    description: str


class ItemCard(BaseModel):
    title: str
    description: str
    imageUrl: Optional[str] = None
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
