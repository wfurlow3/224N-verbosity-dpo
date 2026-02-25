import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

api_key = os.getenv("MOONSHOT_API_KEY")
if api_key is None:
    raise ValueError("MOONSHOT_API_KEY not found")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.ai/v1",
)

# Load first prompt
with open("data/prompts/prompt_pool.jsonl") as f:
    first = json.loads(f.readline())

prompt = first["prompt"]

response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=512,
)

print("PROMPT:\n", prompt)
print("\nRESPONSE:\n", response.choices[0].message.content)