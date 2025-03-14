import uuid
import json
from typing import Optional, List, Dict, Any
import sys

import httpx
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from loguru import logger

# Загружаем переменные окружения
load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ROSPATENT_API_KEY: str
    URL_SEARCH: str
    LOG_LEVEL: str = "DEBUG"  # Для отладки выставляем DEBUG

    class Config:
        env_file = ".env"


settings = Settings()

# Настраиваем loguru
logger.remove()  # удаляем стандартный обработчик
logger.add(sys.stdout, level=settings.LOG_LEVEL)

openai.api_key = settings.OPENAI_API_KEY

app = FastAPI(title="Патентный поиск")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализируем модели для эмбеддингов и QA
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Временные кэши и сессии
conversations: Dict[str, Dict[str, Any]] = {}
patent_cache: Dict[str, Any] = {}


class TextRequest(BaseModel):
    input: str
    conversation_id: Optional[str] = None


class TextResponse(BaseModel):
    conversation_id: str
    refined_query: str
    prompt_used: str
    llm_response: str
    current_match: Dict[str, Any]
    current_index: int
    total_matches: int
    switched: bool


async def refine_search_term(query: str) -> str:
    system_prompt = (
        "Вы являетесь научным помощником, специализирующимся на поиске патентной информации. "
        "Перепишите следующий исходный поисковый запрос так, чтобы он содержал только ключевые термины о чем должен "
        "быть патент и был оптимизирован для поиска патентной информации в базе данных всех патентов, "
        "но не добавляйте лишние слова, которых нет в исходном запросе. Возвращайте только уточнённый запрос без "
        "дополнительных комментариев (НИКОГДА НЕ ИСПОЛЬЗУЙ патенты/патент/технология/технологии). \n\n"
        "Например: Исходный запрос: Интересно узнать про технологию обработки молока. Ответ: обработка молока\n\n"
        "Исходный запрос: Жвачка со сменой вкуса. Ответ: жвательная резинка\n\n"
        "Исходный запрос: Патент на устройство для уборки снега. Ответ: устройство для уборки снега\n\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Исходный запрос: {query}"}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        refined_query = response.choices[0].message.content.strip()
        refined_query = refined_query.replace('патент', '').rstrip()
        # if "патент" not in query.lower():
        #     refined_query = " ".join(word for word in refined_query.split() if word.lower() != "патент")
        logger.debug(f"Уточнённый запрос: {refined_query}")
        return refined_query
    except Exception as e:
        logger.exception("Ошибка при уточнении запроса через LLM")
        return query


async def is_request_for_patent_number_llm(query: str) -> bool:
    system_prompt = (
        "Вы являетесь научным помощником по патентному анализу. "
        "Определите, относится ли следующий запрос к запросу на получение номера патента. "
        "Ответьте только 'True' или 'False' без дополнительных пояснений."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Запрос: {query}"}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=5
        )
        result = response.choices[0].message.content.strip().lower()
        logger.debug(f"Результат проверки запроса на номер патента: {result}")
        return result.startswith("true")
    except Exception as e:
        logger.exception("Ошибка при определении запроса на номер патента через LLM")
        return False


async def is_followup_question(query: str, history: List[Dict[str, str]]) -> bool:
    system_prompt = (
        "Вы являетесь помощником по патентному анализу. "
        "На основе всей истории общения и нового запроса, определите, является ли данный запрос уточняющим вопросом относительно ранее предоставленной патентной информации, или это новый запрос. "
        "Если пользователь пытается уточнить детали ранее найденного патента, ответьте 'True'. "
        "Если это совершенно новый запрос, ответьте 'False'. "
        "Ответьте только 'True' или 'False' без дополнительных пояснений."
    )
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"История общения:\n{history_text}\nНовый запрос: {query}"}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=5
        )
        result = response.choices[0].message.content.strip().lower()
        logger.debug(f"Результат проверки на уточняющий запрос: {result}")
        return result.startswith("true")
    except Exception as e:
        logger.exception("Ошибка при определении уточняющего запроса через LLM")
        return False


def apply_keyword_filter(matches: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    filtered = []
    for m in matches:
        snippet = m.get("snippet", "").lower()
        if all(kw in snippet for kw in keywords):
            filtered.append(m)
    return filtered


async def get_top_matches(query: str, limit: int = 3, sort: str = "publication_date:desc",
                          dataset: str = "ru_since_1994", filter_params: Optional[Dict[str, Any]] = None,
                          original_query: str = None) -> List[Dict[str, Any]]:
    refined_query = query
    cache_key = f"{refined_query}_{limit}_{sort}_{dataset}_{json.dumps(filter_params, sort_keys=True) if filter_params else ''}"
    if cache_key in patent_cache:
        logger.info("Использую кэшированные результаты поиска патентов.")
        return patent_cache[cache_key]

    payload = {
        "q": refined_query,
        "limit": limit * 3,
        "pre_tag": "<b>",
        "post_tag": "</b>"
    }
    if filter_params:
        payload["filter"] = filter_params

    logger.debug(f"Отправляю запрос к API РосПатента с payload: {payload}")
    url = settings.URL_SEARCH
    headers = {
        "Authorization": f"Bearer {settings.ROSPATENT_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=500)
            logger.debug(f"Получен ответ от API: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Ошибка API РосПатента: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"Ошибка API РосПатента: {response.text}")
            data = response.json()
            logger.debug(f"Данные API: {data}")
    except Exception as e:
        logger.exception("Ошибка при обращении к API РосПатента")
        raise HTTPException(status_code=500, detail="Ошибка при обращении к API РосПатента")

    hits = data.get("hits", [])
    logger.debug(f"Найдено {len(hits)} hit'ов в ответе API")
    if not hits:
        raise HTTPException(status_code=404, detail="Результаты не найдены по запросу")

    matches = []
    for hit in hits:
        snippet = hit.get("snippet", {})
        matches.append({
            "id": hit.get("id", ""),
            "title": snippet.get("title", ""),
            "snippet": snippet.get("description", ""),
            "similarity": 0.0
        })
        logger.debug(f"Название патента: {hit.get('title', '')}, Сниппет: {snippet.get('description', '')}")
    logger.debug(f"Сформированы матчи: {[match['id'] for match in matches]}")

    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    logger.debug("Вычислен эмбеддинг запроса")
    snippet_texts = [match["snippet"] for match in matches]
    snippet_embeddings = sbert_model.encode(snippet_texts, convert_to_tensor=True)
    logger.debug("Вычислены эмбеддинги для всех сниппетов")
    similarities = util.cos_sim(query_embedding, snippet_embeddings)
    logger.debug(f"Вычисленные сходства: {[float(similarities[0][i].item()) for i in range(len(matches))]}")
    for i, match in enumerate(matches):
        match["similarity"] = float(similarities[0][i].item())

    sorted_matches = sorted(matches, key=lambda x: x.get("similarity", 0), reverse=True)
    logger.debug(f"Матчи после сортировки по сходству: {[{'id': m['id'], 'sim': m['similarity']} for m in sorted_matches]}")
    if not sorted_matches:
        raise HTTPException(status_code=404, detail="Результаты не найдены по запросу")

    max_similarity = sorted_matches[0]["similarity"]
    if max_similarity < 0.6:
        logger.warning("Максимальная схожесть результатов ниже 0.6, возвращаю лучшие найденные патенты, хотя релевантность может быть низкой.")

    dynamic_threshold = max(0.7, max_similarity * 0.8)
    filtered = [m for m in sorted_matches if m.get("similarity", 0) >= dynamic_threshold]
    logger.debug(f"Порог сходства: {dynamic_threshold:.2f}, отфильтровано: {[{'id': m['id'], 'sim': m['similarity']} for m in filtered]}")

    keywords = []
    if original_query:
        logger.debug(f"Применяю дополнительную фильтрацию по ключевым словам: {keywords}")
        keyword_filtered = apply_keyword_filter(filtered if filtered else sorted_matches, keywords)
        if keyword_filtered:
            logger.debug(f"После фильтрации по ключевым словам осталось: {[m['id'] for m in keyword_filtered]}")
            filtered = keyword_filtered
        else:
            logger.warning("Дополнительная фильтрация по ключевым словам не дала результатов, возвращаю топовые по сходству.")

    if len(filtered) < limit:
        filtered = sorted_matches[:limit]
    else:
        filtered = filtered[:limit]

    patent_cache[cache_key] = filtered
    logger.info(f"Найдено {len(filtered)} патентов после динамической оценки сходства (порог: {dynamic_threshold:.2f}).")
    return filtered


def get_ml_answer(question: str, context: str) -> str:
    try:
        result = qa_pipeline(question=question, context=context)
        logger.debug(f"ML QA pipeline вернул: {result}")
        if result and result.get("score", 0) > 0.3:
            return result.get("answer", "")
    except Exception as e:
        logger.exception("Ошибка при вызове QA pipeline")
    return ""


def generate_prompt(user_query: str, context_match: Dict[str, Any], history: List[Dict[str, str]],
                    ml_answer: str = "") -> str:
    history_text = ""
    if history:
        history_text = "История общения:\n" + "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]
        ) + "\n"
    prompt = (
        f"{history_text}"
        f"Запрос пользователя: {user_query}\n"
        f"Контекст патента:\n{context_match.get('snippet', 'Нет информации')}\n"
    )
    if ml_answer:
        prompt += f"\nПредварительный ML-ответ на вопрос: {ml_answer}\n"
    prompt += (
        "Предоставь формальный, детальный и объективный ответ четко на запрос пользователя, "
        "основываясь на предоставленной информации.\n\n"
        "Если в данных есть информация о патенте, используйте её для ответа, "
        "но НЕ подразумевайте, что пользователь сам упомянул этот патент, "
        "если это явно не указано в его запросе или истории чата."
    )
    logger.debug(f"Сгенерирован prompt для LLM: {prompt[:200]}...")
    return prompt


async def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
    system_prompt = (
        "Вы являетесь экспертом в области патентного анализа с глубокими знаниями технических и юридических аспектов. "
        "Пожалуйста, предоставьте формальный, детальный и объективный ответ на основе предоставленной информации. "
        "Не используйте неформальные выражения, юмор или эмодзи."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        answer = response.choices[0].message.content.strip()
        logger.debug(f"Ответ LLM: {answer[:200]}...")
        return answer
    except Exception as e:
        logger.exception("Ошибка при вызове LLM")
        raise HTTPException(status_code=500, detail=f"Ошибка LLM: {str(e)}")


@app.post("/text", response_model=TextResponse)
async def text_endpoint(request: TextRequest):
    original_input = request.input.strip()
    logger.info(f"Получен запрос: {original_input}")
    refined_query = await refine_search_term(original_input)
    switched = False

    if not request.conversation_id:
        conversation_id = str(uuid.uuid4())
        logger.info(f"Создана новая сессия: {conversation_id}")
        top_matches = await get_top_matches(refined_query, original_query=original_input)
        conversations[conversation_id] = {
            "user_query": refined_query,
            "top_matches": top_matches,
            "current_index": 0,
            "history": [{"role": "user", "content": original_input}]
        }
    else:
        conversation_id = request.conversation_id
        conv = conversations.get(conversation_id)
        if not conv:
            logger.error(f"Сессия {conversation_id} не найдена")
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        conv.setdefault("history", []).append({"role": "user", "content": original_input})
        logger.info(f"Обновление сессии: {conversation_id}")

        if await is_request_for_patent_number_llm(refined_query):
            current_match = conv["top_matches"][conv["current_index"]]
            answer = f"Номер патента: {current_match.get('id', 'не найден')}."
            conv["history"].append({"role": "assistant", "content": answer})
            logger.info("Запрос определён как запрос на номер патента")
            return TextResponse(
                conversation_id=conversation_id,
                refined_query=conv["user_query"],
                prompt_used="—",
                llm_response=answer,
                current_match=current_match,
                current_index=conv["current_index"],
                total_matches=len(conv["top_matches"]),
                switched=switched
            )
        if not await is_followup_question(original_input, conv.get("history", [])):
            logger.info("Обнаружен новый запрос, выполняется новый поиск патентов")
            top_matches = await get_top_matches(refined_query, original_query=original_input)
            conv["top_matches"] = top_matches
            conv["current_index"] = 0

        conv["user_query"] = refined_query

    conv = conversations[conversation_id]
    current_match = conv["top_matches"][conv["current_index"]]
    ml_answer = get_ml_answer(conv["user_query"], current_match.get("snippet", ""))
    prompt = generate_prompt(conv["user_query"], current_match, conv.get("history", []), ml_answer)
    llm_response = await call_llm(prompt)
    conv.setdefault("history", []).append({"role": "assistant", "content": llm_response})
    logger.info(f"Ответ сгенерирован для сессии {conversation_id}")

    return TextResponse(
        conversation_id=conversation_id,
        refined_query=conv["user_query"],
        prompt_used=prompt,
        llm_response=llm_response,
        current_match=current_match,
        current_index=conv["current_index"],
        total_matches=len(conv["top_matches"]),
        switched=switched
    )


@app.get("/health")
async def health_check():
    logger.info("Проверка работоспособности сервиса")
    return {"status": "OK", "message": "Сервис патентного поиска работает корректно."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
