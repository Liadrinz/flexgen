import os
import random

from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from flexygen import Insertable


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 1. 加载Tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


# 2. 使用FlexyGen接口包裹模型
model = Insertable.wrap(model, tokenizer)


# 3. 为模型注入触发器
@model.trigger("emoji")
def emoji_trigger(input_ids) -> bool:
    sentence = tokenizer.batch_decode(input_ids)[0].strip()
    return sentence.endswith(("，", "。", "！", "？"))


# 4. 为模型注入外部调用
@model.invocation("emoji")
def generate_random_emoji(input_ids) -> List[str]:
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
    bsz = input_ids.size(0)
    return [random_emoji() for _ in range(bsz)]


# 5. 生成
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "为什么天空是蓝色的？"},
], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
# 每句话后面都有一个emoji
print(tokenizer.batch_decode(outputs)[0])
