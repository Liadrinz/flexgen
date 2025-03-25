import os
import random

from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from flexygen import SentenceLevelFlexyGen, SentenceLevelGenerationState


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 1. Load HF Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


# 2. Wrap the Model with FlexyGen Interface
model = SentenceLevelFlexyGen.wrap(model, tokenizer)


# 3. Inject a splicer
@model.splicer("prob")
def prob_trigger(state: SentenceLevelGenerationState):
    if state.end_of_sentences[0]:
        if min(state.sentence_token_probs[0]) < 0.1:
            # Returns reflection words when the minimum 
            # token probability of a sentence is lower than 0.1
            return " ... Wait, I'm not sure. "


# 4. Generate
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "Why the sky is blue?"},
], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    do_sample=True,
    max_new_tokens=128,
    return_dict_in_generate=True,
    output_scores=True,
)
print(tokenizer.batch_decode(outputs.sequences)[0])
