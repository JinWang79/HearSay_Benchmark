import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

def load_model(model_path, device="cuda:0"):

    print(f"[Phi4Handler] Loading from: {model_path} on {device}...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        attn_impl = 'flash_attention_2' if torch.cuda.get_device_capability()[0] >= 8 else 'eager'
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype=torch.float16,
            _attn_implementation=attn_impl
        ).eval()
        
        try:
            gen_config = GenerationConfig.from_pretrained(model_path)
        except:
            gen_config = GenerationConfig.from_model_config(model.config)
            
        print("[Phi4Handler] Loaded successfully.")
        return {"model": model, "processor": processor, "gen_config": gen_config, "device": device}
        
    except Exception as e:
        print(f"[Phi4Handler] Load Error: {e}")
        raise e

def run_inference(model_bundle, audio_path, prompt_text):


    model = model_bundle["model"]
    processor = model_bundle["processor"]
    gen_config = model_bundle["gen_config"]
    device = model_bundle["device"]

    try:

        if audio_path is not None:
            audio_array, _ = librosa.load(audio_path, sr=16000)
            
            full_prompt = f'<|user|><|audio_1|>{prompt_text}<|end|><|assistant|>'
            
            audio_input = [audio_array, 16000]
            inputs = processor(
                text=full_prompt, 
                audios=[audio_input], 
                return_tensors='pt'
            ).to(device)
        else:
            full_prompt = f"<|system|>You are a helpful assistant.<|end|><|user|>{prompt_text}<|end|><|assistant|>"
            inputs = processor(
                text=full_prompt, 
                return_tensors='pt'
            ).to(device)
        
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                generation_config=gen_config,
                do_sample=False
            )
            
        response_ids = generate_ids[:, input_len:]
        response = processor.batch_decode(
            response_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()

    except Exception as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return "Error: CUDA OOM"
        return f"Error: Inference Failed ({str(e)[:100]})"