"""Model registry for CNN architectures used in the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Type

import torch


@dataclass
class ModelSpec:
    name: str
    constructor: Callable[..., torch.nn.Module]
    metadata: Dict[str, object] = field(default_factory=dict)


_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(name: str, constructor: Callable[..., torch.nn.Module], metadata: Optional[Dict[str, object]] = None) -> None:
    key = name.lower()
    if key in _REGISTRY:
        raise ValueError(f"Model '{name}' already registered")
    _REGISTRY[key] = ModelSpec(name=key, constructor=constructor, metadata=metadata or {})


def get_model(name: str, **kwargs) -> torch.nn.Module:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    spec = _REGISTRY[key]
    default_args = spec.metadata.get("default_args", {})
    params = {**default_args, **kwargs}
    return spec.constructor(**params)


def list_models() -> List[str]:
    return sorted(_REGISTRY.keys())


def iter_model_specs() -> Iterable[ModelSpec]:
    return _REGISTRY.values()
