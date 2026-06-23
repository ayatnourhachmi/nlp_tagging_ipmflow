"""Groq and OpenAI LLM providers for zero-shot classification."""

import os

from groq import Groq
from openai import OpenAI

from ipmflow.llm.prompts import SYSTEM_PROMPT, build_user_prompt

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")


def call_groq(
    text: str,
    rules_context: str | None = None,
    horizon_context: str | None = None,
) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, rules_context, horizon_context)},
        ],
        max_tokens=1000,
        temperature=0,
    )
    return response.choices[0].message.content


def call_openai(
    text: str,
    rules_context: str | None = None,
    horizon_context: str | None = None,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, rules_context, horizon_context)},
        ],
        max_tokens=1000,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


PROVIDERS = {
    "groq": {"fn": call_groq, "model": GROQ_MODEL, "label": "Groq LLM"},
    "openai": {"fn": call_openai, "model": OPENAI_MODEL, "label": "OpenAI LLM"},
}
