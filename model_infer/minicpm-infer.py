import os
import base64
import time
from openai import OpenAI


VLM_BASE_URL = os.getenv("VLM_BASE_URL")
VLM_MODEL_ID = "/home/research/xxx" 

def encode_audio_uri(audio_path):
    if not os.path.exists(audio_path): return None
    ext = os.path.splitext(audio_path)[1].lower().replace(".", "")
    if ext not in ["wav", "mp3", "flac"]: ext = "wav"
    mime = "mpeg" if ext == "mp3" else ext 
    
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:audio/{mime};base64,{b64}"

def load_model(model_path=None, device=None): 

    api_key="token-xxxx"
    url = VLM_BASE_URL
    print(f"[MiniCPMHandler] Connecting to vLLM at {url}...")
    
    client = OpenAI(api_key=api_key, base_url=url)
    return client

def run_inference(client, audio_path, prompt_text):

    data_uri = encode_audio_uri(audio_path)
    if not data_uri: return "Error: Audio not found"

    try:
        response = client.chat.completions.create(
            model=VLM_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": data_uri}
                        }
                    ]
                }
            ],

            extra_body={"stop_token_ids": [151645, 151643]},
            max_tokens=512,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        err = str(e)
        if "Connection refused" in err: return "Error: vLLM Service Down"
        return f"Error: API Failed ({err[:100]})"