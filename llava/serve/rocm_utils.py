"""
llava/serve/rocm_utils.py

Lightweight ROCm/HIP detection and configuration helpers.
Intentionally has no LLaVA-model imports so it can be unit-tested in
isolation without instantiating the full model stack.
"""

import os

import torch


def is_rocm() -> bool:
    """Return True when the active PyTorch build is backed by ROCm/HIP."""
    return getattr(torch.version, "hip", None) is not None


def configure_hip_allocator() -> bool:
    """
    Set PYTORCH_HIP_ALLOC_CONF to reduce HIP memory-allocator fragmentation
    that can cause segfaults during large-model inference on ROCm 5.x.

    Uses os.environ.setdefault so a pre-existing user-supplied value is
    never overwritten.

    Returns True if the variable was set by this call, False if it was
    already present.
    """
    _DEFAULT = "garbage_collection_threshold:0.8,max_split_size_mb:512"
    prev = os.environ.get("PYTORCH_HIP_ALLOC_CONF")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _DEFAULT)
    return prev is None
