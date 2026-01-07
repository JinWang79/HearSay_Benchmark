import os
import sys
import torch


try:
    from kimia_infer.api.kimia import KimiAudio
except ImportError:
    raise ImportError("Failed to import 'kimia_infer'.")


KIMI_SAMPLING_PARAMS = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.1,  
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

def load_model(model_path, device="cuda"):

    print(f"[KimiHandler] Loading model from: {model_path} ...")
    
    try:
        model = KimiAudio(model_path=model_path, load_detokenizer=False)
        print("[KimiHandler] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[KimiHandler] Error loading model: {e}")
        raise e

def run_inference(model, audio_path, prompt_text):

    if not os.path.exists(audio_path):
        return f"Error: Audio file not found: {audio_path}"

    messages = [
        {"role": "user", "message_type": "text", "content": prompt_text},
        {"role": "user", "message_type": "audio", "content": audio_path}
    ]

    try:
        _, text_output = model.generate(
            messages, 
            **KIMI_SAMPLING_PARAMS, 
            output_type="text"
        )
        return text_output.strip()
        
    except Exception as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return "Error: CUDA OOM"
        return f"Error: Inference Failed ({str(e)[:100]})"