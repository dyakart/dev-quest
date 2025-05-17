from __future__ import annotations

import json
import os
import re
from typing import List, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# ─────────────────────────── Константы ────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")

DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 6))

app = FastAPI(title="DevQuest AI Backend", version="1.2.0")


# ─────────────────────────── Pydantic схемы ───────────────────────
class QuestionRequest(BaseModel):
    language: Literal["python", "javascript", "typescript", "go", "java", "c++"]
    difficulty: int = Field(..., ge=0, le=10)


class QuestionResponse(BaseModel):
    question: str
    options: List[str]
    correct_answer_index: int
    explanation: str


# ─────────────────────────── Промпт ───────────────────────────────

def difficulty_to_label(level: int) -> str:
    if level == 0:
        return "очень лёгкий"
    if 1 <= level <= 4:
        return "junior"
    if 5 <= level <= 7:
        return "middle"
    return "senior"


def build_prompt(language: str, level: int) -> str:
    """Формирует промпт и добавляет случайный UID, чтобы LLM не повторялась."""
    label = difficulty_to_label(level)
    uid = os.urandom(4).hex()  # 8-символьный случайный идентификатор
    return (
        f"Ты опытный интервьюер по {language}. Пиши только на русском.\n"
        f"Сгенерируй ОДИН совершенно новый вопрос (не повторяй предыдущие) с четырьмя вариантами ответа для уровня {label}.\n"
        "Верни ТОЛЬКО JSON-объект со следующими ключами: question, options (4 шт), correct_answer_index, explanation (не пустое).\n"
        f"UID запроса: {uid}"
    )


# ─────────────────────────── Вспомогательные ──────────────────────
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
RU_LETTERS = {"А": 0, "Б": 1, "В": 2, "Г": 3}


def russian_ratio(text: str) -> float:
    return len(CYRILLIC_RE.findall(text)) / len(text) if text else 0.0


def extract_json(text: str) -> Optional[dict]:
    clean = re.sub(r"```[a-zA-Z]*\n?", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    if m := re.search(r"{.*?}", clean, re.DOTALL):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_plain_text(text: str) -> Optional[dict]:
    qm = re.search(r"(?:Question|Вопрос)[:\s]+(.+?)\n", text, re.IGNORECASE)
    if not qm:
        return None
    question = qm.group(1).strip()

    opts = [m.group(1).strip() for m in re.finditer(r"[A-DА-Г1-4][).:»\-]?\s*(.+)", text)]
    if len(opts) != 4:
        return None

    if m := re.search(r"Answer[:\s]+([A-DА-Г1-4])", text, re.IGNORECASE):
        tok = m.group(1).upper()
        idx = int(tok) - 1 if tok.isdigit() else RU_LETTERS.get(tok, ord(tok) - ord("A"))
    else:
        return None

    explanation = ""
    if ex := re.search(r"Explanation[:\s]+(.+)$", text, re.IGNORECASE | re.DOTALL):
        explanation = ex.group(1).strip()

    return {
        "question": question,
        "options": opts,
        "correct_answer_index": idx,
        "explanation": explanation,
    }


def is_valid(data: dict) -> bool:
    try:
        obj = QuestionResponse(**data)  # type: ignore[arg-type]
    except Exception:
        return False
    if len(obj.options) != 4 or not obj.explanation.strip() or not 0 <= obj.correct_answer_index < 4:
        return False
    return russian_ratio(" ".join([obj.question] + obj.options + [obj.explanation])) >= 0.5


# ─────────────────────────── LLM вызовы ───────────────────────────
async def call_openrouter(messages: list[dict], model: str | None = None, with_format: bool = True) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/devquest",
        "X-Title": "DevQuest",
    }
    payload = {
        "model": model or OPENROUTER_MODEL,
        "temperature": 0.05,
        "max_tokens": 1024,  # ограничиваем, чтобы не превышать кредит
        "messages": messages,
    }
    if with_format:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(OPENROUTER_URL, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(502, r.text)

    data = r.json()
    if "choices" not in data:
        # Некорректный ответ OpenRouter (часто содержит {'error':{...}})
        raise HTTPException(502, f"OpenRouter bad response: {data}")
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(502, f"Unexpected LLM schema: {data}") from e


async def ask_llm(prompt: str) -> dict:
    sys = {"role": "system", "content": "You are a helpful assistant."}
    usr = {"role": "user", "content": prompt}
    last = ""

    for attempt in range(1, MAX_RETRIES + 1):
        msgs = [sys, usr]
        if attempt > 1:
            missing = [k for k in ("question", "options", "correct_answer_index", "explanation") if
                       k not in (extract_json(last) or {})]
            msgs.append({
                "role": "user",
                "content": "‼️ Предыдущий ответ неполный. Добавь: " + ", ".join(
                    missing) + ". Верни ТОЛЬКО JSON.\n" + last[:800],
            })
        # Первая попытка с response_format, далее без него (на случай, если модель не поддерживает)
        last = await call_openrouter(msgs, with_format=attempt == 1)
        data = extract_json(last) or parse_plain_text(last)
        if data and is_valid(data):
            return data

    # Fallback на gpt‑4o‑mini (если уже использовался, пропустим)
    if OPENROUTER_MODEL != "gpt-4o-mini":
        last = await call_openrouter([sys, usr], model="gpt-4o-mini", with_format=False)
        data = extract_json(last) or parse_plain_text(last)
        if data and is_valid(data):
            return data

    raise HTTPException(502, f"LLM failed after retries. Last raw: {last[:1200]}")


# ─────────────────────────── FastAPI route ────────────────────────
@app.post("/generate_question", response_model=QuestionResponse, tags=["AI"])
async def generate_question(req: QuestionRequest) -> QuestionResponse:  # type: ignore[valid-type]
    prompt = build_prompt(req.language, req.difficulty)
    data = await ask_llm(prompt)
    return QuestionResponse(**data)
