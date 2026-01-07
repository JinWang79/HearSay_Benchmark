import os
import sys
import torch
import librosa

try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
except ImportError:
    pass 

def load_model(model_path, device="cuda:0"):

    print(f"[QwenOmniHandler] Loading from: {model_path} ...")
    
    if model_path not in sys.path:
        sys.path.append(model_path)
        
    try:
        
        from qwen_omni_utils import process_mm_info
        
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else "auto"
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device
        )
        
        print("[QwenOmniHandler] Loaded successfully.")
        
        return {
            "model": model, 
            "processor": processor, 
            "utils_func": process_mm_info, 
            "device": device
        }
        
    except ImportError:
        raise ImportError(f"Failed to import 'qwen_omni_utils'.")
    except Exception as e:
        print(f"[QwenOmniHandler] Load Error: {e}")
        raise e

def run_inference(model_bundle, audio_path, prompt_text):


    model = model_bundle["model"]
    processor = model_bundle["processor"]
    process_mm_info = model_bundle["utils_func"]
    
    if audio_path is not None:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt_text}],
            },
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_path}],
            },
        ]
    else:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]    

    USE_AUDIO_IN_VIDEO = True  

    try:

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        inputs = processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        
        inputs = inputs.to(model.device).to(model.dtype)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            text_ids = model.generate(
                **inputs, 
                return_audio=False, 
                max_new_tokens=256
            )
        
        generated_ids = text_ids[:, input_len:]
        output_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()

    except Exception as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return "Error: CUDA OOM"
        return f"Error: Inference Failed ({str(e)[:100]})"