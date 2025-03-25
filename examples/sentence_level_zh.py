import os
import random

from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from flexygen import SentenceLevelFlexyGen, SentenceLevelGenerationState


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 1. 加载Tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


# 2. 使用SentenceLevelFlexyGen接口包裹模型
model = SentenceLevelFlexyGen.wrap(model, tokenizer)


# 3. 为模型注入连接器
@model.splicer("prob")
def prob_trigger(state: SentenceLevelGenerationState):
    if state.end_of_sentences[0]:
        if min(state.sentence_token_probs[0]) < 0.1:
            # 如果句子中token的最小概率小于0.1
            # 则返回反思的话语
            return "等等，我不太确定。"


# 4. 生成
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "为什么天空是蓝色的？"},
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
