import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from .config import settings
from .supported_models import resolve_model_id


@dataclass(frozen=True)
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModel
    device: torch.device


def _get_hf_token() -> Optional[str]:
    token = os.getenv("HF_TOKEN")
    if not token:
        return None
    return token


def resolve_device() -> torch.device:
    device_name = settings.device.lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name in {"cpu", "cuda"}:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EMBEDDINGS_DEVICE is set to cuda but no CUDA device is available")
        return torch.device(device_name)
    raise RuntimeError("EMBEDDINGS_DEVICE must be one of: auto, cpu, cuda")


@lru_cache(maxsize=settings.max_loaded_models)
def _load_model_bundle(resolved_model_id: str) -> ModelBundle:
    token = _get_hf_token()
    device = resolve_device()
    _configure_cuda_memory_fraction(device)

    bnb_kwargs = _resolve_bitsandbytes_kwargs()

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_id,
        token=token,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        resolved_model_id,
        token=token,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        **bnb_kwargs,
    )

    _ensure_padding_token(tokenizer, model)

    if "device_map" not in bnb_kwargs and device.type != "cpu":
        model.to(device)
    if "device_map" in bnb_kwargs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return ModelBundle(tokenizer=tokenizer, model=model, device=device)


def load_model_bundle(model_id: str) -> ModelBundle:
    resolved_model_id = resolve_model_id(model_id)
    return _load_model_bundle(resolved_model_id)


@lru_cache(maxsize=1)
def _configure_cuda_memory_fraction(device: torch.device) -> None:
    if device.type != "cuda":
        return
    fraction = settings.cuda_memory_fraction
    if fraction is None:
        return
    if not torch.cuda.is_available():
        return
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)


def _ensure_padding_token(tokenizer: AutoTokenizer, model: AutoModel) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    if tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


def _resolve_bitsandbytes_kwargs() -> dict:
    mode = settings.bitsandbytes
    if mode is None:
        return {}
    if not torch.cuda.is_available():
        raise RuntimeError("EMBEDDINGS_BITSANDBYTES requires CUDA")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "bitsandbytes is not installed. Install it to use EMBEDDINGS_BITSANDBYTES."
        )
    if mode == "8bit":
        return {"load_in_8bit": True, "device_map": "auto"}
    if mode == "4bit":
        return {"load_in_4bit": True, "device_map": "auto"}
    return {}
