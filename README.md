# FlexGen

FlexGen is a easy-to-use tool for **controllable generation** of Large Language Models (LLMs). It lets you easily wrap popular transformer models to control the generation process based on custom criteria. This means you can create interactive, flexible applications that adjust responses on the fly, making text generation smarter and more responsive.

## Get Started

### Traditional Generation using HF models

```python
from flexgen import Interruptible
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import seed_everything


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", quantization_config=BitsAndBytesConfig(load_in_4bit=True))

# Generate using the original model
input_text = tokenizer.apply_chat_template([
    {"role": "user", "content": "Why the sky is blue?"},
], tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
seed_everything(42)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
print("====== Generation using original model ======")
print(tokenizer.batch_decode(outputs)[0])
"""
<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>Why the sky is blue? It has to do with the light that is scattered... but I also remember that the atmosphere is made up of molecules that absorb and reflect light. So, if the sky is blue because the light that is scattered is blue, but the molecules themselves are red, so why does the sky appear blue?

Wait, I also remember that the sky is blue because the light that is scattered in the atmosphere is blue. But if the molecules are red, then when light hits them, it can't see the blue. So, how is the sky blue? Is it because the light is absorbed and not scattered, or because it's scattered but with
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
"""

# Wrap the model with `Interruptible`
Interruptible.wrap(model, tokenizer)


# Register interrupt criteria
@model.register_interrupt_criteria("test")
def test_criteria(
    input_ids,
    scores,
    raw_logits,
    decoder_attentions,
    cross_attentions,
    decoder_hidden_states
) -> bool:
    return tokenizer.batch_decode(input_ids)[0].strip().endswith("I also remember")  # When "I also remember" is generated, trigger interruption


# Register interrupt invocation
@model.register_interrupt_invocation("test")
def invoke_test(
    input_ids,
    scores,
    raw_logits,
    decoder_attentions,
    cross_attentions,
    decoder_hidden_states
):
    bsz = input_ids.size(0)
    return [" my physics teacher telling me that" for _ in range(bsz)]  # When interruption triggered, force to generate " my physics teacher telling me that"


# Generate using the wrapped model
seed_everything(42)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
print("====== Generation using interruptable model ======")
print(tokenizer.batch_decode(outputs)[0])
"""
<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>Why the sky is blue? It has to do with the light that is scattered... but I also remember my physics teacher telling me that it's because the light that is scattered... Wait, that's not exactly correct. Wait, maybe it's because the light that is scattered... I'm getting confused here. Could you explain why the sky is blue? I think it's because the light that is scattered... but I also remember my physics teacher telling me that it's because the light that is scattered... but that's not exactly correct. Wait, maybe it's because the light that is scattered... but I'm getting confused here. I need to figure this out.
"""
```
