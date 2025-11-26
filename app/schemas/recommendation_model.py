from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class KakaoUser(BaseModel):
    id: str
    type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class KakaoRequest(BaseModel):
    timezone: str
    params: Dict[str, str] = {}
    block: Dict[str, str]
    utterance: str
    lang: Optional[str] = None
    user: KakaoUser

class KakaoAction(BaseModel):
    id: str
    name: str
    params: Dict[str, str] = {}
    detailParams: Optional[Dict[str, Any]] = None
    clientExtra: Optional[Dict[str, Any]] = None

class KakaoIntent(BaseModel):
    id: str
    name: str
    extra: Optional[Dict[str, Any]] = None

class KakaoSkillRequest(BaseModel):
    intent: KakaoIntent
    userRequest: KakaoRequest
    bot: Optional[Dict[str, Any]] = None
    action: KakaoAction
    contexts: Optional[List[Dict[str, Any]]] = []

class SimpleText(BaseModel):
    text: str

class ListItem(BaseModel):
    title: str
    description: str

class ItemCardHead(BaseModel):
    title: str
    description: Optional[str] = None

class ItemCard(BaseModel):
    head: Optional[ItemCardHead] = None
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
