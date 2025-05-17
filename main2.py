import json
import os
import re
import asyncio

from typing import List, Literal
from pydantic import BaseModel, Field

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv  # Переменные окружения

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("API ключ для OPENAI не установлен!")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="DevQuest AI Backend", version="0.0.0.1")


# ────────────────────────────────────────────────────────────────────────────────
# Pydantic схемы
# ────────────────────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    """Запрос от фронтенда."""

    language: Literal[
        "python",
        "javascript",
        "typescript",
        "go",
        "java",
        "c++",
    ] = Field(..., description="Язык программирования для задания")
    difficulty: int = Field(
        ..., ge=0, le=10, description="Сложность: 0 (очень легко) … 10 (senior)"
    )


class QuestionResponse(BaseModel):
    """Схема, которую возвращаем на фронт"""

    question: str
    options: List[str]
    correct_answer_index: int
    explanation: str


def difficulty_to_label(level: int) -> str:
    """Определяет уровень сложности задачи и возвращает строку для промпта"""

    if level == 0:
        return "очень лёгкий"
    if 1 <= level <= 4:
        return "junior"
    if 5 <= level <= 7:
        return "middle"
    return "senior"  # 8‑10


def build_prompt(language: str, level: int) -> str:
    """Формирует текст запроса к GPT, учитывая язык и уровень сложности."""

    label = difficulty_to_label(level)
    return (
        f"Ты опытный интервьюер по {language}. "
        f"Сгенерируй ОДИН вопрос с четырьмя вариантами ответа для разработчика уровня {label}.\n"
        "Верни ТОЛЬКО объект JSON со следующими ключами:\n"
        "  - question: string (формулировка вопроса)\n"
        "  - options: array из 4 string (варианты ответа)\n"
        "  - correct_answer_index: integer 0‑3 (номер правильного варианта)\n"
        "  - explanation: string (пояснение правильного ответа)\n"
        "Никакого дополнительного текста, описаний или markdown — только JSON."
    )


def _ask_gpt_sync(prompt: str) -> dict:
    """Синхронный вызов OpenAI ChatCompletion (SDK v1)."""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```json|```$", "", content).strip()
    return json.loads(content)


async def ask_gpt(prompt: str) -> dict:  # noqa: D401 (Not docstring style issue)
    """Асинхронная обёртка над синхронным _ask_gpt_sync через to_thread."""

    try:
        return await asyncio.to_thread(_ask_gpt_sync, prompt)
    except json.JSONDecodeError as err:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from AI: {err}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/generate_question",
    response_model=QuestionResponse,
    tags=["AI"],
    summary="Сгенерировать тестовый вопрос",
)
async def generate_question(req: QuestionRequest) -> QuestionResponse:  # type: ignore[valid-type]
    """Эндпоинт для генерации одного тестового вопроса."""

    prompt = build_prompt(req.language, req.difficulty)
    data = await ask_gpt(prompt)

    # Минимальная валидация структуры
    expected_keys = {"question", "options", "correct_answer_index", "explanation"}
    if not expected_keys.issubset(data):
        raise HTTPException(status_code=502, detail="AI response missing required keys")
    if not isinstance(data["options"], list) or len(data["options"]) != 4:
        raise HTTPException(status_code=502, detail="'options' must be an array of 4 strings")

    return data  # FastAPI сконвертирует в QuestionResponse
