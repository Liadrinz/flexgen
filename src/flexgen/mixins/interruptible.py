import torch

from collections import OrderedDict
from typing import Union
from torch import nn
from transformers.generation.utils import (
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)


class Interruptible(GenerationMixin):
    
    def __init__(self):
        self.interrupt_criteria_dict = OrderedDict({})
        self.interrupt_invocation_dict =  OrderedDict({})
    
    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        return tokenizer
    
    def register_interrupt_criteria(self, name):
        def decorate(func):
            self.interrupt_criteria_dict[name] = func
            return func
        return decorate
    
    def register_interrupt_invocation(self, name):
        def decorate(func):
            self.interrupt_invocation_dict[name] = func
            return func
        return decorate
    
    def _get_interrupt_names(self):
        return self.interrupt_criteria_dict.keys()
    
    def _invoke_interrupt(
        self, 
        input_ids,
        scores,
        raw_logits,
        decoder_attentions,
        cross_attentions,
        decoder_hidden_states,
    ):
        names = self._get_interrupt_names()
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        for key in names:
            if not self.interrupt_criteria_dict[key](input_ids, scores, raw_logits, decoder_attentions, cross_attentions, decoder_hidden_states):
                continue
            invoke_fn = self.interrupt_invocation_dict.get(key, None)
            if invoke_fn is None:
                continue
            invoked_results = invoke_fn(input_ids, scores, raw_logits, decoder_attentions, cross_attentions, decoder_hidden_states)
            outputs = self.tokenizer(invoked_results, return_tensors="pt", padding=True, add_special_tokens=False).to(input_ids.device)
            outputs = {k: v for k, v in outputs.items() if k in [input_ids_key, "attention_mask"]}
            return outputs
        return None
    
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            
            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()
            next_token_logits = next_token_logits.to(input_ids.device)
            
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
            
            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            
            # invocation
            invocation = self._invoke_interrupt(
                input_ids,
                scores,
                raw_logits,
                decoder_attentions,
                cross_attentions,
                decoder_hidden_states,
            )
            if invocation is not None:
                input_ids = torch.cat([input_ids, invocation["input_ids"]], dim=-1)
                model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], invocation["attention_mask"]], dim=1)
                max_cache_position = model_kwargs["cache_position"][-1]
                extra_cache_position = torch.arange(max_cache_position + 1, max_cache_position + invocation["input_ids"].size(1) + 1, dtype=torch.long, device=input_ids.device)
                model_kwargs["cache_position"] = torch.cat([model_kwargs["cache_position"], extra_cache_position], dim=0)
                cur_len += invocation["input_ids"].size(1)

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    @staticmethod
    def wrap(model: GenerationMixin, tokenizer) -> "Interruptible":
        Interruptible.__init__(model)
        model.register_tokenizer = lambda *args, **kwargs: Interruptible.register_tokenizer(model, *args, **kwargs)
        model.register_interrupt_criteria = lambda *args, **kwargs: Interruptible.register_interrupt_criteria(model, *args, **kwargs)
        model.register_interrupt_invocation = lambda *args, **kwargs: Interruptible.register_interrupt_invocation(model, *args, **kwargs)
        model._get_interrupt_names = lambda *args, **kwargs: Interruptible._get_interrupt_names(model, *args, **kwargs)
        model._invoke_interrupt = lambda *args, **kwargs: Interruptible._invoke_interrupt(model, *args, **kwargs)
        model._sample = lambda *args, **kwargs: Interruptible._sample(model, *args, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.register_tokenizer(tokenizer)
        return model
