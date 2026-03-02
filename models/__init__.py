"""Convenience imports for model registry and implementations."""

from .registry import register_model, get_model, list_models, iter_model_specs

# Import model modules to trigger registration side effects.
from . import basic  # noqa: F401
from . import advanced  # noqa: F401

__all__ = [
    "register_model",
    "get_model",
    "list_models",
    "iter_model_specs",
]
