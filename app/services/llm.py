import random
import time

from google import genai
from google.genai import types


gemini_client = None


def configure_gemini_client(api_key):
    global gemini_client
    try:
        gemini_client = genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to configure Gemini SDK: {e}")


GEMINI_MODELS = [
    ("gemini-2.5-flash", 60),
    ("gemini-flash-latest", 60),
]


def get_gemini_response_with_retry(prompt: str, max_retries: int = 3) -> str:
    last_error = None
    for model_name, timeout in GEMINI_MODELS:
        for attempt in range(max_retries):
            try:
                print(f"[*] [{model_name}] Attempt {attempt + 1}/{max_retries}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=10000,
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "ProjectIdeas": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "ProjectName": {"type": "string"},
                                            "Description": {"type": "string"},
                                            "SkillsPracticed": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["ProjectName", "Description", "SkillsPracticed"]
                                    },
                                    "minItems": 2,
                                    "maxItems": 2
                                }
                            },
                            "required": ["ProjectIdeas"]
                        }
                    )
                )
                if not response or not response.text:
                    raise ValueError("Empty response received from Gemini.")
                
                
                json_string = response.text.strip()
                if json_string.startswith("```json"):
                    json_string = json_string[7:]
                elif json_string.startswith("```"):
                    json_string = json_string[3:]
                if json_string.endswith("```"):
                    json_string = json_string[:-3]
                    
                return json_string.strip()
            
            except Exception as e:
                last_error = str(e)
                print(f"[*] [{model_name}] Attempt {attempt + 1} failed: {last_error[:100]}...")
                
                if attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                    print(f"[*] Overloaded. Waiting {sleep_time:.2f}s before retry...")
                    time.sleep(sleep_time)
        
        print(f"[*] Exhausted all retries for {model_name}. Moving to next fallback model...")

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")
