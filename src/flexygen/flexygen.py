import torch

from transformers.generation.streamers import BaseStreamer
from typing import List, Union, Optional
from .mixins.base import TriggerInjectableMixin, InvokeType
from .data import TriggerResult, TriggerResultList, GenerationState


class FlexyGen(TriggerInjectableMixin):
    
    def _check_and_proc_trigger_results(self, trigger_results: List, state: GenerationState) -> List[TriggerResult]:
        bsz = state.input_ids.size(0)
        if isinstance(trigger_results, list):
            if len(trigger_results) > 1:
                try:
                    assert len(trigger_results) == bsz
                except:
                    raise ValueError(f"Expect the splicer to return a list of size {bsz}, but got {len(trigger_results)}")
            else:
                trigger_results = trigger_results * bsz
        else:
            trigger_results = [trigger_results]
        trigger_results = TriggerResult.batch_cast(trigger_results)
        return trigger_results
    
    def _invoke(
        self,
        invoke_type: InvokeType,
        state: GenerationState,
    ) -> List[TriggerResult]:
        bsz = state.input_ids.size(0)
        overall_invoke_results = [TriggerResult() for _ in range(bsz)]
        trigger = self.get_trigger_names(invoke_type)
        for key in trigger:
            trigger_fn = self.trigger_dict[invoke_type][key]
            state = self._proc_state_before_trigger(state, invoke_type, key)
            trigger_results: TriggerResultList = trigger_fn(state)
            trigger_results = self._proc_results_after_trigger(trigger_results, invoke_type, key)
            trigger_results = self._check_and_proc_trigger_results(trigger_results, state)
            for i in range(bsz):
                overall_invoke_results[i] += trigger_results[i]
        return overall_invoke_results
    
    def _proc_state_before_trigger(self, state: GenerationState, invoke_type: InvokeType, key: str) -> GenerationState:
        return state
    
    def _proc_results_after_trigger(self, trigger_results: TriggerResultList, invoke_type: InvokeType, key: str) -> TriggerResultList:
        return trigger_results
    
    def _postprocess_invocation(
        self,
        invocation_results: List[TriggerResult],
        state: GenerationState,
        streamer: Optional[BaseStreamer] = None,
    ) -> GenerationState:
        tensor_results = self.tokenizer(
            [r.sentence for r in invocation_results],
            return_tensors="pt",
            add_special_tokens=False,
        )
        if tensor_results["input_ids"].size(1) > 0:
            # Put invocation results to stream somewhere else
            if streamer is not None:
                for col in range(tensor_results["input_ids"].size(1)):
                    streamer.put(tensor_results["input_ids"][:, col].cpu())
            
            state.input_ids = torch.cat([state.input_ids, tensor_results["input_ids"]], dim=-1)
            state.model_kwargs["attention_mask"] = torch.cat([state.model_kwargs["attention_mask"], tensor_results["attention_mask"]], dim=1)
            
            max_cache_position = state.model_kwargs["cache_position"][-1]
            extra_cache_position = torch.arange(max_cache_position + 1, max_cache_position + tensor_results["input_ids"].size(1) + 1, dtype=torch.long, device=state.input_ids.device)
            state.model_kwargs["cache_position"] = torch.cat([state.model_kwargs["cache_position"], extra_cache_position], dim=0)
            state.current_length += tensor_results["input_ids"].size(1)
        
        return state
    
    def on_each_iteration_end(
        self,
        state: GenerationState,
        streamer: Optional[BaseStreamer] = None,
    ) -> GenerationState:
        splicer_invoke_results = self._invoke(InvokeType.SPLICER, state)
        return self._postprocess_invocation(splicer_invoke_results, state, streamer)

    @classmethod
    def wrap(cls, model, tokenizer) -> "FlexyGen":
        super().wrap(model, tokenizer)
        model._invoke = lambda *args, **kwargs: cls._invoke(model, *args, **kwargs)
        model._check_and_proc_trigger_results = lambda *args, **kwargs: cls._check_and_proc_trigger_results(model, *args, **kwargs)
        model._proc_state_before_trigger = lambda *args, **kwargs: cls._proc_state_before_trigger(model, *args, **kwargs)
        model._proc_results_after_trigger = lambda *args, **kwargs: cls._proc_results_after_trigger(model, *args, **kwargs)
        model._postprocess_invocation = lambda *args, **kwargs: cls._postprocess_invocation(model, *args, **kwargs)
        return model
