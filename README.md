# FlexyGen

English | [中文版](README-zh.md)

FlexyGen is an easy-to-use **LLM controllable generation tool** that controls and adjusts model generation content by injecting **triggers** and **invocations** into the model.

## Installation

```shell
pip install flexygen
```

## Getting Started

Demo Code: [examples/emoji.py](examples/emoji.py)

### 0. Import Dependencies

```python
import random

from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from flexygen import Insertable
```

### 1. Load HF Tokenizer and Model

```python
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)
```

### 2. Wrap the Model with FlexyGen Interface

The `Insertable` interface can be used to insert content into the sentence being generated when certain conditions are met.

```python
model = Insertable.wrap(model, tokenizer)
```

### 3. Inject a Trigger

Inject a trigger named `emoji` into the model.

During the model generation process, the trigger is called every time a token is generated, and an invocation is triggered when the trigger returns `True`.

Here `input_ids` is the content currently generated by the model.

```python
@model.trigger("emoji")
def emoji_trigger(input_ids) -> bool:
    sentence = tokenizer.batch_decode(input_ids)[0].strip()
    return sentence.endswith((".", "!", "?"))
```

### 4. Inject an Invocation

Injects an invocation named `emoji`.

This invocation is triggered when the trigger of the same name returns `True`.

The string returned by the call is inserted after the currently generated content (Currently only `batch_size=1` supported).

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

### 5. Generate

```python
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "Why the sky is blue?"},
], tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
# Emoji attached after each sentence.
print(tokenizer.batch_decode(outputs)[0])
```
