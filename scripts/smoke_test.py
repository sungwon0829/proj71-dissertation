from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, sys

model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda")

messages = [{"role": "user", "content": "In one sentence, what is cognitive behavioural therapy?"}]
inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")

out = model.generate(**inputs, max_new_tokens=60)
print(tok.decode(out[0], skip_special_tokens=True))
print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")