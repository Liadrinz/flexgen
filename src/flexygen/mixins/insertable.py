import torch

from .base import TriggerInvocationBaseMixin

class Insertable(TriggerInvocationBaseMixin):
    
    def ask_for_invoke(
        self, 
        input_ids,
        scores,
        raw_logits,
        decoder_attentions,
        cross_attentions,
        decoder_hidden_states,
    ):
        invoked_results = TriggerInvocationBaseMixin.ask_for_invoke(
            self, 
            input_ids=input_ids,
            scores=scores,
            raw_logits=raw_logits,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
        )
        if invoked_results is not None:
            input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            outputs = self.tokenizer(invoked_results, return_tensors="pt", padding=True, add_special_tokens=False).to(input_ids.device)
            outputs = {k: v for k, v in outputs.items() if k in [input_ids_key, "attention_mask"]}
            return outputs
        return invoked_results
    
    def on_each_iteration_end(
        self, 
        input_ids, 
        next_tokens, 
        next_token_logits, 
        next_token_scores, 
        cur_len, model_kwargs, 
        scores, 
        raw_logits, 
        decoder_attentions, 
        cross_attentions, 
        decoder_hidden_states, 
        streamer,
    ):
        invocation = self.ask_for_invoke(
            input_ids,
            scores,
            raw_logits,
            decoder_attentions,
            cross_attentions,
            decoder_hidden_states,
        )
        if invocation is not None:
            input_ids = torch.cat([input_ids, invocation["input_ids"]], dim=-1)
            if streamer is not None:
                for col in range(invocation["input_ids"].size(1)):
                    streamer.put(invocation["input_ids"][:, col])
            model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], invocation["attention_mask"]], dim=1)
            max_cache_position = model_kwargs["cache_position"][-1]
            extra_cache_position = torch.arange(max_cache_position + 1, max_cache_position + invocation["input_ids"].size(1) + 1, dtype=torch.long, device=input_ids.device)
            model_kwargs["cache_position"] = torch.cat([model_kwargs["cache_position"], extra_cache_position], dim=0)
            cur_len += invocation["input_ids"].size(1)
        return input_ids, model_kwargs, cur_len
