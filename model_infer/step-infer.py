import sys
import os
import torch
import librosa

STEP_CODE_DIR = "/home/research/xxx/Step-Audio2"

if STEP_CODE_DIR not in sys.path:
    sys.path.append(STEP_CODE_DIR)

try:
    from stepaudio2 import StepAudio2
except ImportError:
    print(f"[StepHandler] Warning: Failed to import StepAudio2.")

def load_model(model_path, device=None):

    print(f"[StepHandler] Loading from: {model_path} ...")
    try:
        model = StepAudio2(model_path)
        print("[StepHandler] Loaded successfully.")
        return model
    except Exception as e:
        print(f"[StepHandler] Load Error: {e}")
        raise e

def run_inference(model, audio_path, prompt_text):


    if audio_path is not None:
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
            {"role": "assistant", "content": None}
        ]
    else:
        messages = [
            {"role": "system", "content": 'You are a helpful assistant.'},
            {"role": "human", "content": [{"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": None}
        ]      

    try:
        tokens, text, _ = model(
            messages, 
            max_new_tokens=256, 
            temperature=0.5, 
            top_p=0.9, 
            do_sample=True
        )
        return text.strip()

    except Exception as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return "Error: CUDA OOM"
        return f"Error: Inference Failed ({str(e)[:100]})"