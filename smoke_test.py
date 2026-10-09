

from pathlib import Path
from dotenv import dotenv_values
from google import genai

env_path = Path(__file__).resolve().parent / "APIKEY.env"
config = dotenv_values(env_path)
api_key = config.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(f"GEMINI_API_KEY not found in {env_path}")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Say hello in one sentence.",
)

print(response.text)
