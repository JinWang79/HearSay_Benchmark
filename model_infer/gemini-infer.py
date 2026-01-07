import os
import base64
import time
from openai import OpenAI


OPENROUTER_URL = "https://openrouter.ai/api/v1"

def encode_audio_base64(audio_path):

    if not os.path.exists(audio_path): return None, None
    ext = os.path.splitext(audio_path)[1].lower().replace(".", "")
    if ext not in ["wav", "mp3", "flac", "ogg"]: ext = "wav"
    
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return b64, ext

def load_model(api_key=None, model_id="google/gemini-2.5-flash-preview-09-2025"):

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("[GeminiHandler] Missing API Key.")

    print(f"[GeminiHandler] Initializing Client for {model_id}...")
    client = OpenAI(base_url=OPENROUTER_URL, api_key=key, timeout=120.0)
    

    return {"client": client, "model_id": model_id}

def run_inference(model_bundle, audio_path, prompt_text):

    client = model_bundle["client"]
    model_id = model_bundle["model_id"]
    
    raw_b64, fmt = encode_audio_base64(audio_path)
    if not raw_b64: return "Error: Audio not found"

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "input_audio", 
                            "input_audio": {"data": raw_b64, "format": fmt}
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=512
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        err = str(e)
        if "429" in err:
            time.sleep(2)
            return "Error: Rate Limit"
        if "401" in err:
            return "Error: Unauthorized"
        return f"Error: API Failed ({err[:100]})"