import os
import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def load_model(model_path, device="cuda:0"):

    print(f"[Qwen2Handler] Loading from: {model_path} on {device}...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True
        ).eval()
        
        print("[Qwen2Handler] Loaded successfully.")
        return {"model": model, "processor": processor, "device": device}
        
    except Exception as e:
        print(f"[Qwen2Handler] Load Error: {e}")
        raise e

def run_inference(model_bundle, audio_path, prompt_text):

    if not os.path.exists(audio_path):
        return f"Error: Audio not found {audio_path}"

    model = model_bundle["model"]
    processor = model_bundle["processor"]
    
    try:
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': [
                {'type': 'audio', 'audio_url': audio_path}, 
                {'type': 'text', 'text': prompt_text}
            ]}
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        audio_array, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
        
        inputs = processor(
            text=[text], 
            audio=[audio_array], 
            return_tensors="pt", 
            padding=True
        )
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    except Exception as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return "Error: CUDA OOM"
        return f"Error: Inference Failed ({str(e)[:100]})"