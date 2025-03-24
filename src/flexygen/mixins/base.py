from collections import OrderedDict
from transformers.generation.utils import GenerationMixin
from .callback import GenerationMixinWithPerIterCallbacks
from ..utils import filter_arguments


class TriggerInvocationBaseMixin(GenerationMixinWithPerIterCallbacks):
    
    def __init__(self):
        self.trigger_dict = OrderedDict({})
        self.invocation_dict =  OrderedDict({})
    
    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        return tokenizer
    
    def trigger(self, name):
        def decorate(func):
            self.trigger_dict[name] = filter_arguments(func)
            return func
        return decorate
    
    def invocation(self, name):
        def decorate(func):
            self.invocation_dict[name] = filter_arguments(func)
            return func
        return decorate
    
    def get_trigger_names(self):
        return self.trigger_dict.keys()
    
    def ask_for_invoke(self, *args, **kwargs):
        names = self.get_trigger_names()
        for key in names:
            if not self.trigger_dict[key](*args, **kwargs):
                continue
            invoke_fn = self.invocation_dict.get(key, None)
            if invoke_fn is None:
                continue
            return invoke_fn(*args, **kwargs)
        return None
    
    @classmethod
    def wrap(cls, model: GenerationMixin, tokenizer) -> "TriggerInvocationBaseMixin":
        cls.__init__(model)
        
        # TODO: Fix bugs in iterative function replacement
        # for name in dir(cls):
        #     if not name.startswith("__"):
        #         exec(f"model.{name} = lambda *args, **kwargs: cls.{name}(model, *args, **kwargs)", {"cls": cls, "model": model})
        
        # Hard-encoded function replacement
        model.register_tokenizer = lambda *args, **kwargs: cls.register_tokenizer(model, *args, **kwargs)
        model.trigger = lambda *args, **kwargs: cls.trigger(model, *args, **kwargs)
        model.invocation = lambda *args, **kwargs: cls.invocation(model, *args, **kwargs)
        model.get_trigger_names = lambda *args, **kwargs: cls.get_trigger_names(model, *args, **kwargs)
        model.ask_for_invoke = lambda *args, **kwargs: cls.ask_for_invoke(model, *args, **kwargs)
        model.on_each_iteration_end = lambda *args, **kwargs: cls.on_each_iteration_end(model, *args, **kwargs)
        model._dola_decoding = lambda *args, **kwargs: cls._dola_decoding(model, *args, **kwargs)
        model._contrastive_search = lambda *args, **kwargs: cls._contrastive_search(model, *args, **kwargs)
        model._sample = lambda *args, **kwargs: cls._sample(model, *args, **kwargs)
        model._beam_search = lambda *args, **kwargs: cls._beam_search(model, *args, **kwargs)
        model._group_beam_search = lambda *args, **kwargs: cls._group_beam_search(model, *args, **kwargs)
        model._constrained_beam_search = lambda *args, **kwargs: cls._constrained_beam_search(model, *args, **kwargs)
        model._assisted_decoding = lambda *args, **kwargs: cls._assisted_decoding(model, *args, **kwargs)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.register_tokenizer(tokenizer)
        return model
