import os
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from flexygen import FlexyGen, GenerationState


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 1. Load HF Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


# 2. Wrap the Model with FlexyGen Interface
model = FlexyGen.wrap(model, tokenizer)


# 3. Inject a splicer trigger
@model.splicer("emoji")
def emoji_trigger(state: GenerationState) -> bool:
    def random_emoji():
        ranges = [
            (0x1F600, 0x1F64F),
            (0x1F300, 0x1F5FF),
            (0x1F680, 0x1F6FF),
            (0x2700, 0x27BF),
        ]
        start, end = random.choice(ranges)
        code_point = random.randint(start, end)
        return chr(code_point)
    sentence = tokenizer.batch_decode(state.input_ids)[0].strip()
    if sentence.endswith((",", ".", "!", "?")):
        return random_emoji()  # Returns a random emoji character


# 4. Generate
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "Why the sky is blue?"},
], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
# Emoji attached after each sentence.
print(tokenizer.batch_decode(outputs)[0])
