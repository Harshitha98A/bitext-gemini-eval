import os
from typing import Literal
import time

from google import genai
from dotenv import load_dotenv
from google.genai.errors import ClientError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=api_key)

ModelName = Literal["gemini_main", "gemini_alt", "gemini_tuned"]

# For now, leave tuned model empty; we'll fill this after B2.
TUNED_MODEL_ID = None  # e.g. "tunedModels/support-qa-sri-v1"


def call_model(model_name: ModelName, question: str) -> str:
    """
    Call different Gemini configs depending on model_name.
    Right now:
      - gemini_main: precise, concise style
      - gemini_alt: slightly more friendly style
      - gemini_tuned: (later) your tuned model
    """
    if model_name == "gemini_main":
        model_id = "gemini-2.5-flash"
        system_prompt = (
            "You are a precise, concise customer support assistant. "
            "Answer in one short paragraph, focusing only on the policy."
        )
    elif model_name == "gemini_alt":
        model_id = "gemini-2.5-flash"
        system_prompt = (
            "You are a friendly customer support assistant. "
            "Answer clearly in 2–3 short sentences, staying on-topic."
        )
    elif model_name == "gemini_tuned":
        if not TUNED_MODEL_ID:
            raise RuntimeError("TUNED_MODEL_ID is not set for gemini_tuned")
        model_id = TUNED_MODEL_ID
        system_prompt = "You are a specialized assistant for this company’s customer service."
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    full_prompt = f"{system_prompt}\n\nQuestion: {question}"

    max_retries = 3

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=full_prompt,
            )
            # If it succeeds, return immediately
            return (resp.text or "").strip()

        except ClientError as e:
            # Check if it's a rate limit / quota error (429 RESOURCE_EXHAUSTED)
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_seconds = 30
                print(
                    f"[WARN] Rate limit for {model_id}. "
                    f"Sleeping {wait_seconds}s (attempt {attempt+1}/{max_retries})"
                )
                time.sleep(wait_seconds)
                # then retry
                continue
            else:
                # Some other client error we don't want to swallow
                raise

    # If we exhaust all retries without success, raise an error
    raise RuntimeError(f"Failed to call Gemini after {max_retries} attempts.")
