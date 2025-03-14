import os
import re
import uuid
from typing import Optional, List, Dict, Any

import httpx
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

conversations: Dict[str, Dict[str, Any]] = {}


class TextRequest(BaseModel):
    input: str
    conversation_id: Optional[str] = None


class TextResponse(BaseModel):
    conversation_id: str
    prompt_used: str
    llm_response: str
    current_match: Dict[str, Any]
    current_index: int
    total_matches: int
    switched: bool


def preprocess_query(query: str) -> str:
    q = query.lower()
    patterns = [
        r"^(привет|здравствуйте|добрый день)[,!\s]*",
        r"я хочу узнать[,!\s]*",
        r"найди(,)?\s*пожалуйста[,!\s]*",
        r"пожалуйста[,!\s]*"
    ]
    for pattern in patterns:
        q = re.sub(pattern, "", q)
    return q.strip(" ,.!?;:")


async def extract_search_term(query: str) -> str:
    prompt = (
        f"Извлеки основное ключевое слово для поиска патентов из следующего запроса: \"{query}\". "
        "Ответь только одним словом."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ответ должен содержать только одно слово."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        term = response.choices[0].message.content.strip()
        return term if term else query
    except Exception as e:
        return query


async def get_top_matches(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    url = os.getenv("URL_SEARCH")
    payload = {"qn": query, "limit": limit}
    headers = {
        "Authorization": f"Bearer {os.getenv('ROSPATENT_API_KEY')}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=500)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ошибка API РосПатента: {response.text}"
            )
        data = response.json()

    hits = data.get("hits", [])
    matches = []
    for hit in hits:
        snippet = hit.get("snippet", {})
        match = {
            "id": hit.get("id", ""),
            "title": snippet.get("title", ""),
            "snippet": snippet.get("description", "")
        }
        matches.append(match)
    if not matches:
        raise HTTPException(status_code=404, detail="Не найдены результаты по запросу от РосПатента")
    return matches


def generate_prompt(user_query: str, context_match: Dict[str, Any]) -> str:
    return (
        f"Запрос пользователя: {user_query}\n"
        f"Контекст патента:\n{context_match['snippet']}\n"
        f"Название патента: {context_match['title']}\n"
        f"ID патента: {context_match['id']}\n"
        "Пожалуйста, сформируйте подробный ответ на основе контекста патента."
    )


async def call_llm(prompt: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Вы являетесь экспертом в патентном анализе. Помо"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при вызове LLM: {str(e)}")


async def check_switch_context(user_message: str, current_match: Dict[str, Any]) -> bool:
    prompt = (
        f"Дан запрос пользователя: \"{user_message}\"\n"
        f"и контекст патента: \"{current_match['snippet']}\"\n"
        "Определите, выражает ли данный запрос намерение перейти к другому патенту "
        "(то есть, пользователь не хочет общаться с данным патентом). "
        "Если да, ответьте только 'True', иначе — 'False'."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ответ должен быть только 'True' или 'False'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        decision_text = response.choices[0].message.content.strip()
        return decision_text.lower().startswith("true")
    except Exception as e:
        return False


@app.post("/text", response_model=TextResponse)
async def text_endpoint(request: TextRequest):
    original_input = request.input.strip()
    processed_query = preprocess_query(original_input)
    switched = False

    if not request.conversation_id:
        search_term = await extract_search_term(processed_query)
        conversation_id = str(uuid.uuid4())
        top_matches = await get_top_matches(search_term)
        conversations[conversation_id] = {
            "user_query": processed_query,
            "top_matches": top_matches,
            "current_index": 0
        }
    else:
        conversation_id = request.conversation_id
        conv = conversations.get(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        current_match = conv["top_matches"][conv["current_index"]]
        if await check_switch_context(processed_query, current_match):
            conv["current_index"] += 1
            if conv["current_index"] >= len(conv["top_matches"]):
                conv["current_index"] = 0
            switched = True
        else:
            conv["user_query"] = processed_query

    conv = conversations[conversation_id]
    current_match = conv["top_matches"][conv["current_index"]]
    prompt = generate_prompt(conv["user_query"], current_match)
    llm_response = await call_llm(prompt)

    return TextResponse(
        conversation_id=conversation_id,
        prompt_used=prompt,
        llm_response=llm_response,
        current_match=current_match,
        current_index=conv["current_index"],
        total_matches=len(conv["top_matches"]),
        switched=switched
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
