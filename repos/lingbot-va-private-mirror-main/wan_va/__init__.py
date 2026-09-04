# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

__all__ = ["configs", "distributed", "modules"]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)
