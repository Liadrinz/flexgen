from dataclasses import asdict
from .flexygen import FlexyGen
from .data import SentenceLevelGenerationState
from copy import deepcopy


class SentenceLevelFlexyGen(FlexyGen):
    
    def _get_max_splitter_length(self, splitters) -> int:
        splitter_batch = self.tokenizer(splitters, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"]
        return splitter_batch.size(1)
    
    def _reset_before_generation(self):
        self.sentence_tokens = None
        self.sentence_token_probs = None
    
    def _proc_state_before_trigger(self, state, invoke_type, key) -> SentenceLevelGenerationState:
        bsz = state.input_ids.size(0)
        if self.sentence_tokens is None:
            self.sentence_tokens = [[] for _ in range(bsz)]
            self.sentence_token_probs = [[] for _ in range(bsz)]
        options = self.trigger_options_dict[invoke_type][key]
        splitters = options.get("splitters", [".", "。", "?", "？", "!", "！"])
        if "_max_splitter_len" not in options:
            options["_max_splitter_len"] = self._get_max_splitter_length(splitters)
        max_splitter_len = options["_max_splitter_len"]
        texts = self.tokenizer.batch_decode(state.input_ids[:, -max_splitter_len:])
        end_of_sentence = [any([t.endswith(s) for s in splitters]) for t in texts]
        for i, flag in enumerate(end_of_sentence):
            next_token = state.next_tokens[i].item()
            self.sentence_tokens[i].append(next_token)
            self.sentence_token_probs[i].append(state.next_token_scores[i].softmax(dim=-1)[next_token].item())
        state = SentenceLevelGenerationState(
            **asdict(state),
            end_of_sentences=end_of_sentence,
            sentence_tokens=deepcopy(self.sentence_tokens),
            sentence_token_probs=deepcopy(self.sentence_token_probs),
        )
        for i, flag in enumerate(end_of_sentence):
            if flag:
                self.sentence_tokens[i].clear()
                self.sentence_token_probs[i].clear()
        return state

    @classmethod
    def wrap(cls, model, tokenizer) -> "SentenceLevelFlexyGen":
        super().wrap(model, tokenizer)
        model._invoke = lambda *args, **kwargs: cls._invoke(model, *args, **kwargs)
        model._get_max_splitter_length = lambda *args, **kwargs: cls._get_max_splitter_length(model, *args, **kwargs)
        return model
