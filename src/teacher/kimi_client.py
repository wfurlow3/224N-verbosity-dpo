"""
Kimi (Moonshot) API client for teacher model calls.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "moonshot-v1-8k" # this is the teacher model we'll use to generate our candidates according to different variants. 


def call_kimi(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
) -> str:
    api_key = os.getenv("MOONSHOT_API_KEY")
    if api_key is None:
        raise ValueError("MOONSHOT_API_KEY not found in environment")
    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL) # use openAI to call Kimi
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    if not response.choices[0].message.content:
        raise ValueError("No response from Kimi")
    return response.choices[0].message.content
