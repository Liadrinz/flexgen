# FlexyGen

中文版 | [English](README.md)

FlexyGen是一款易用的**LLM可控生成工具**，通过控制反转（IoC）的思想，允许开发者向模型中注入一系列**触发器**实现对模型生成过程的控制。在触发器中可以根据模型当前的生成状态对生成内容进行修改（目前仅支持在当前生成句子后拼接新的内容）。

## 安装

```shell
pip install flexygen
```

## 入门

示例代码：[examples/emoji_zh.py](examples/emoji_zh.py)

### 0. 导入依赖

```python
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from flexygen import FlexyGen, GenerationState
```

### 1. 加载Tokenizer和模型

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

### 2. 使用FlexyGen接口包裹模型

`FlexyGen`接口可以在满足一定条件时向正在生成的句子中插入内容。

```python
model = FlexyGen.wrap(model, tokenizer)
```

### 3. 为模型注入拼接触发器

为模型注入一个名为`emoji`的拼接触发器。

模型每生成一个token都将调用该触发器。

如果当前句子以`"，", "。", "！", "？"`结尾，则触发器返回一个随机emoji字符，该字符会被拼接到当前句子的后面，否则返回`None`，且不会改变当前句子。

`state`中存储着生成状态，可以通过`state.input_ids`访问当前句子。

```python
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
```

### 4. 生成

```python
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "为什么天空是蓝色的？"},
], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
# 每句话后面都有一个emoji
print(tokenizer.batch_decode(outputs)[0])
```

## 访问`state`中的其他内容

`state`中除了`state.input_ids`还可以访问其他内容：

- `state.model_kwargs`: 模型参数，包含传给模型的`attention_mask`, `past_key_values`等参数
- `state.current_length`: 当前句子长度（token数）
- `state.next_tokens`: 模型当前生成的token，等价于`state.input_ids[:, -1]`
- `state.next_token_logits`: 当前token的logits
- `state.next_token_scores`: 当前token的scores（经过logit_processor处理后的logits）

如果在`generate()`方法中指定`return_dict_in_generate=True`和`output_scores=True`，则触发器中可以访问`state.scores`，即所有token的scores。

如果在`generate()`方法中指定`return_dict_in_generate=True`和`output_logits=True`，则触发器中可以访问`state.raw_logits`，即模型的注意力权重。

如果在`generate()`方法中指定`return_dict_in_generate=True`和`output_attentions=True`，则触发器中可以访问`state.decoder_attentions`, `state.cross_attentions`，分别为解码器的注意力权重和交叉注意力权重（仅encoder-decoder架构）

如果在`generate()`方法中指定`return_dict_in_generate=True`和`output_hidden_states=True`，则触发器中可以访问`state.decoder_hidden_states`，表示解码器每个Transformer层输出的隐状态向量。

## 使用`SentenceLevelFlexyGen`

在一些应用中需要根据句子的概率或句子中某些token的概率来触发某些调用，例如自适应**RAG**会判断句子中token的概率来决定是否触发检索。

此时可以使用`SentenceLevelFlexyGen`：

```python

# ...省略模型定义

from flexygen import SentenceLevelFlexyGen, SentenceLevelGenerationState


model = SentenceLevelFlexyGen.wrap(model, tokenizer)


@model.splicer("prob")
def prob_trigger(state: SentenceLevelGenerationState):
    if state.end_of_sentences[0]:
        if min(state.sentence_token_probs[0]) < 0.1:
            # 如果句子中token的最小概率小于0.1
            # 则返回表示不确定的话语
            return "等等，我不太确定。"


# ...省略生成
```

使用`SentenceLevelFlexyGen`时，触发器中的`state`变为`SentenceLevelGenerationState`对象，比`GenerationState`多了一些内容：

- `state.end_of_sentences: List[bool]`：布尔变量列表，表示一个batch中的每个输出是否生成了一个完整的句子（默认情况下，以`。？！.?!`这六个字符结尾表示生成了一个句子）
- `state.sentence_tokens: List[List[int]]`：当前句子
- `state.sentence_token_probs: List[List[int]]`：当前句子中每个token的概率
