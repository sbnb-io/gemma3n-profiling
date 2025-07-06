from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
import torch.profiler
import time

# Profiling Google Gemma 3n Model Using PyTorch Profiler
# https://github.com/sbnb-io/gemma3n-profiling

MODEL_ID = "google/gemma-3n-e2b-it"
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
PROFILE_TRACE_PATH = "gemma3n-profiling.json"
MAX_NEW_TOKENS = 10

def load_model(model_id):
    return Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    ).eval()

def load_processor(model_id):
    return AutoProcessor.from_pretrained(model_id)

def build_messages(image_url):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]

def prepare_inputs(processor, messages, device):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device, dtype=torch.bfloat16)

def profile_generation(model, processor, inputs, input_len, max_new_tokens, trace_path):
    decoded = ""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=0),
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(2):
            with torch.inference_mode():
                prof.step()
                generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                output_tokens = generation[0][input_len:]
                decoded = processor.decode(output_tokens, skip_special_tokens=True)
    prof.export_chrome_trace(trace_path)
    return decoded

def main():
    model = load_model(MODEL_ID)
    processor = load_processor(MODEL_ID)
    messages = build_messages(IMAGE_URL)
    inputs = prepare_inputs(processor, messages, model.device)
    input_len = inputs["input_ids"].shape[-1]
    decoded = profile_generation(model, processor, inputs, input_len, MAX_NEW_TOKENS, PROFILE_TRACE_PATH)
    print(decoded)

if __name__ == "__main__":
    main()
