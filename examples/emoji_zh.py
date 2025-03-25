import os
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from flexygen import FlexyGen, GenerationState


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 1. 加载Tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


# 2. 使用FlexyGen接口包裹模型
model = FlexyGen.wrap(model, tokenizer)


# 3. 为模型注入连接器
@model.splicer("emoji")
def emoji_splicer(state: GenerationState) -> bool:
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
    if sentence.endswith(("，", "。", "！", "？")):
        return random_emoji()  # 返回一个随机的emoji表情字符


# 4. 生成
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "为什么天空是蓝色的？"},
], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
# 每句话后面都有一个emoji
print(tokenizer.batch_decode(outputs)[0])
