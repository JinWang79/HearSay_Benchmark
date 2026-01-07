import os
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

def load_model(model_path, device="cuda:0"):

    print(f"[MERaLiONHandler] Loading from: {model_path} on {device}...")
    
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            use_safetensors=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16
        ).to(device)
        
        print("[MERaLiONHandler] Loaded successfully.")
        
        return {"model": model, "processor": processor, "device": device}
        
    except Exception as e:
        print(f"[MERaLiONHandler] Load Error: {e}")
        raise e

def run_inference(model_bundle, audio_path, prompt_text):

    if not os.path.exists(audio_path):
        return f"Error: Audio not found {audio_path}"

    model = model_bundle["model"]
    processor = model_bundle["processor"]
    device = model_bundle["device"]

    template = "Given the following audio context: <SpeechHere>\n\nText instruction: {query}"
    final_content = template.format(query=prompt_text)
    
    conversation = [
        [{"role": "user", "content": final_content}]
    ]
    
    chat_prompt = processor.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    try:
        audio_array, _ = librosa.load(audio_path, sr=16000)
        audio_array = [audio_array]*2
        
        inputs = processor(text=chat_prompt, audios=audio_array)
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = inputs[key].to(device)
                if value.dtype == torch.float32:
                    inputs[key] = inputs[key].to(torch.bfloat16)
                    
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        input_len = inputs['input_ids'].size(1)
        generated_ids = outputs[:, input_len:]
        
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    except Exception as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return "Error: CUDA OOM"
        return f"Error: Inference Failed ({str(e)[:100]})"