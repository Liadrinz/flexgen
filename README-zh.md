# FlexGen

中文版 | [English](README.md)

FlexGen是一款易用的**LLM可控生成工具**，通过向模型中注入**触发器**和**外部调用**实现对模型生成内容的控制和调整。

## 入门

示例代码：[examples/emoji_zh.py](examples/emoji_zh.py)

### 0. 导入依赖

```python
import random

from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from flexgen import Insertable
```

### 1. 加载Tokenizer和模型

```python
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)
```

### 2. 使用FlexGen接口包裹模型

`Insertable`接口可以在满足一定条件时向正在生成的句子中插入内容。

```python
model = Insertable.wrap(model, tokenizer)
```

### 3. 为模型注入触发器

注入名为`emoji`的触发器。

模型生成过程中，每生成一个token都会调用触发器，触发器返回`True`时会触发外部调用。

这里`input_ids`是模型当前生成的内容。

```python
@model.trigger("emoji")
def emoji_trigger(input_ids) -> bool:
    sentence = tokenizer.batch_decode(input_ids)[0].strip()
    return sentence.endswith(("。", "！", "？"))
```

### 4. 为模型注入外部调用

注入名为`emoji`的外部调用。

当同名的触发器返回`True`时，将触发这一调用。

调用返回的字符串会被插入到当前生成的内容后面。（目前仅支持`batch_size=1`）

```python
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
```

### 5. 生成

```python
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "为什么天空是蓝色的？"},
], tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
# 每句话后面都有一个emoji
print(tokenizer.batch_decode(outputs)[0])
```
