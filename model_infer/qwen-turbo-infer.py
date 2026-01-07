import os
import base64
import time
from openai import OpenAI


API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
MODEL_NAME = "qwen-omni-turbo"

def encode_audio_uri(audio_path):

    if not os.path.exists(audio_path): return None, None
    ext = os.path.splitext(audio_path)[1].lower().replace(".", "")
    if ext not in ["wav", "mp3", "flac"]: ext = "wav"
    
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:audio/{ext};base64,{b64}", ext

def load_model(path=None, device=None):  

    key = API_KEY
    url = DASHSCOPE_BASE_URL
    
    if not key:
        raise ValueError("[Qwen3Handler] Missing API Key. Set DASHSCOPE_API_KEY env var.")

    print(f"[Qwen3Handler] Initializing Client (URL: {url})...")
    client = OpenAI(api_key=key, base_url=url)
    return client

def run_inference(client, audio_path, prompt_text):

    data_uri, fmt = encode_audio_uri(audio_path)
    if not data_uri: return "Error: Audio not found"

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": data_uri,
                                "format": fmt,
                            }
                        },
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": False},
            temperature=0.1
        )

        full_text = ""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    full_text += delta.content
        
        time.sleep(0.5)
        
        return full_text.strip()

    

    except Exception as e:
        err = str(e)
        if "429" in err:
            time.sleep(2) 
            return "Error: Rate Limit"
        if "400" in err:
            return "Error: Bad Request (Check Audio Format)"
        return f"Error: API Call Failed ({err[:100]})"