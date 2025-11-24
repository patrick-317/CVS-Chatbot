# CVS-Chatbot

편의점 꿀조합 추천 챗봇 API

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
uvicorn main:app --reload
```

## API 문서

서버 실행 후 아래 주소에서 확인 가능:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### POST /api/v1/recommend

편의점 꿀조합 추천

**요청**
```json
{
  "keyword": "스트레스 폭발",
  "category": "디저트"
}
```

**응답**
```json
{
  "combo_name": "악마의 유혹 디저트",
  "main_products": "브라우니",
  "sub_products": "악마빙수, 아메리카노"
}
```
