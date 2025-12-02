# CVS Honey Combo Chatbot 🍯

카카오톡 챗봇에서  
사용자의 자연어 입력을 바탕으로 **CU 편의점 꿀조합**을 추천해주는 FastAPI 기반 백엔드입니다.

- 실제 꿀조합 데이터(`combination.csv`)
- LLM이 생성한 확장 꿀조합(`synthetic_honey_combos_1000.csv`)
- CU 공식 상품 데이터(`cu_official_products.csv`)

위 3개 데이터에서 **실제 판매 상품만 매칭**해서 조합을 만들어 줍니다.

---

## 🚀 실행 방법

```bash
git clone https://github.com/patrick-317/CVS-Chatbot.git
cd CVS-Chatbot

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# .env 에 OPENAI_API_KEY 등 환경변수 설정
cp .env.example .env
# .env 수정

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```
---

POST /api/v1/kakao/recommend

요청 바디 예시
<details> <summary><strong>펼치기 / 접기</strong></summary>
{
  "intent": {
    "id": "test_intent_1",
    "name": "꿀조합추천",
    "extra": {}
  },
  "userRequest": {
    "timezone": "Asia/Seoul",
    "params": {},
    "block": {
      "id": "block_id_1",
      "name": "recommend_block"
    },
    "utterance": "비도 오고 꿀꿀하네",
    "lang": "ko",
    "user": {
      "id": "user_1234",
      "type": "accountId",
      "properties": {
        "appUserId": "user_1234"
      }
    }
  },
  "bot": {
    "id": "bot_1234",
    "name": "CVS_HoneyCombo_Bot"
  },
  "action": {
    "id": "action_1",
    "name": "recommend",
    "params": {},
    "detailParams": {}
  }
}
</details>

---

응답 예시
<details> <summary><strong>펼치기 / 접기</strong></summary>
curl -X 'POST' \
  'http://IP주소:8000/api/v1/kakao/recommend' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "intent": {
    "id": "test_intent_1",
    "name": "꿀조합추천",
    "extra": {}
  },
  "userRequest": {
    "timezone": "Asia/Seoul",
    "params": {},
    "block": {
      "id": "block_id_1",
      "name": "recommend_block"
    },
    "utterance": "비도 오고 꿀꿀하네",
    "lang": "ko",
    "user": {
      "id": "user_1234",
      "type": "accountId",
      "properties": {
        "appUserId": "user_1234"
      }
    }
  },
  "bot": {
    "id": "bot_1234",
    "name": "CVS_HoneyCombo_Bot"
  },
  "action": {
    "id": "action_1",
    "name": "recommend",
    "params": {},
    "detailParams": {}
  }
}
</details>
'
