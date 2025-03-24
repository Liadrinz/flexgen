import random
import numpy as np
import torch
import os
import inspect


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def filter_arguments(func):
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = sig.parameters
        
        # 处理位置参数
        var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        if not var_positional:
            positional = [p for p in params.values() if p.kind in (inspect.Parameter.POSITIONAL_ONLY, 
                                                                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            args = args[:len(positional)]  # 截断多余位置参数

        # 处理关键字参数
        var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not var_keyword:
            allowed_kwargs = set()
            positional_names = [p.name for p in params.values() if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            
            # 添加可接受的关键字参数
            for name in positional_names[len(args):]:  # 未被位置参数填充的
                allowed_kwargs.add(name)
            allowed_kwargs.update(  # 添加keyword-only参数
                p.name for p in params.values() if p.kind == inspect.Parameter.KEYWORD_ONLY
            )
            
            # 过滤无效关键字参数
            kwargs = {k: v for k, v in kwargs.items() 
                    if k in allowed_kwargs and k not in positional_names[:len(args)]}

        return func(*args, **kwargs)
    return wrapper
