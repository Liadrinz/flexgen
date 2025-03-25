import torch

from dataclasses import dataclass, field
from typing import List, Any, Iterable, Optional, Union, Dict, Tuple


@dataclass
class TriggerResult:
    
    sentence: str = ""

    @classmethod
    def batch_cast(cls, container: Iterable[Any]) -> List["TriggerResult"]:
        result = []
        for i, element in enumerate(container):
            if isinstance(element, cls):
                result.append(element)
            elif isinstance(element, str):
                result.append(cls(sentence=element))
            elif element is None:
                result.append(cls())
            else:
                raise ValueError(f"Expect all results to be of type `InvocationResult`, `str` or `NoneType`, but got `{type(element)}` at index {i}.")
        return result

    def __add__(self, other: "TriggerResult") -> "TriggerResult":
        return TriggerResult(self.sentence + other.sentence)
    
    def __iadd__(self, other: "TriggerResult") -> "TriggerResult":
        self.sentence += other.sentence
        return self

    def __or__(self, other: "TriggerResult") -> "TriggerResult":
        return TriggerResult(f"{self.sentence}|{other.sentence}")
    
    def __ior__(self, other: "TriggerResult") -> "TriggerResult":
        self.sentence += "|" + other.sentence
        return self


@dataclass
class GenerationState:
    
    input_ids: torch.LongTensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    current_length: int = 0
    
    next_tokens: torch.LongTensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    next_token_logits: torch.FloatTensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float))
    next_token_scores: torch.FloatTensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float))
    
    scores: Optional[Tuple[torch.FloatTensor]] = None
    raw_logits: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    def autoregressive_props(self):
        return self.input_ids, self.model_kwargs, self.current_length


@dataclass
class SentenceLevelGenerationState(GenerationState):
    
    end_of_sentences: List[bool] = field(default_factory=list)
    sentence_tokens: List[List[int]] = field(default_factory=list)
    sentence_token_probs: List[List[float]] = field(default_factory=list)


TriggerOutput = Optional[Union[TriggerResult, str]]
TriggerResultList = Union[TriggerOutput, List[TriggerOutput]]
