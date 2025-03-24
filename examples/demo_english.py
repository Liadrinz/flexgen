import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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

# Wrap the model
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
):
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
