from enum import Enum
from typing import Union, List, Callable
from collections import OrderedDict
from transformers.generation.utils import GenerationMixin
from .generation import GenerationMixinWithPerIterCallbacks
from ..utils import filter_arguments


class InvokeType(Enum):
    
    SPLICER = 1


class TriggerInjectableMixin(GenerationMixinWithPerIterCallbacks):
    
    def __init__(self):
        self.trigger_dict = {k: OrderedDict({}) for k in InvokeType}
        self.trigger_options_dict = {k: OrderedDict({}) for k in InvokeType}
    
    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        return tokenizer
    
    def splicer(self, name, **options):
        def decorate(func: Callable[..., Union[List[Union[bool, str]], Union[bool, str]]]):
            self.trigger_dict[InvokeType.SPLICER][name] = filter_arguments(func)
            self.trigger_options_dict[InvokeType.SPLICER][name] = options
            return func
        return decorate
    
    def get_trigger_names(self, invoke_type: InvokeType):
        return self.trigger_dict[invoke_type].keys()
    
    @classmethod
    def wrap(cls, model: GenerationMixin, tokenizer) -> "TriggerInjectableMixin":
        cls.__init__(model)
        
        # TODO: Fix bugs in iterative function replacement
        # for name in dir(cls):
        #     if not name.startswith("__"):
        #         exec(f"model.{name} = lambda *args, **kwargs: cls.{name}(model, *args, **kwargs)", {"cls": cls, "model": model})
        
        # Hard-encoded function replacement
        model.register_tokenizer = lambda *args, **kwargs: cls.register_tokenizer(model, *args, **kwargs)
        model.splicer = lambda *args, **kwargs: cls.splicer(model, *args, **kwargs)
        model.get_trigger_names = lambda *args, **kwargs: cls.get_trigger_names(model, *args, **kwargs)
        model.on_each_iteration_end = lambda *args, **kwargs: cls.on_each_iteration_end(model, *args, **kwargs)
        model._dola_decoding = lambda *args, **kwargs: cls._dola_decoding(model, *args, **kwargs)
        model._contrastive_search = lambda *args, **kwargs: cls._contrastive_search(model, *args, **kwargs)
        model._sample = lambda *args, **kwargs: cls._sample(model, *args, **kwargs)
        model._beam_search = lambda *args, **kwargs: cls._beam_search(model, *args, **kwargs)
        model._group_beam_search = lambda *args, **kwargs: cls._group_beam_search(model, *args, **kwargs)
        model._constrained_beam_search = lambda *args, **kwargs: cls._constrained_beam_search(model, *args, **kwargs)
        model._assisted_decoding = lambda *args, **kwargs: cls._assisted_decoding(model, *args, **kwargs)
        model._reset_before_generation = lambda *args, **kwargs: cls._reset_before_generation(model, *args, **kwargs)
        model.generate = lambda *args, **kwargs: cls.generate(model, *args, **kwargs)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.register_tokenizer(tokenizer)
        return model
