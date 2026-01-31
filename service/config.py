import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

SERVICE_VERSION = "0.1.0"


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc


def _get_env_optional_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Invalid {name} value: {value}")


@dataclass(frozen=True)
class Settings:
    device: str
    cuda_memory_fraction: Optional[float]
    bitsandbytes: Optional[str]
    max_loaded_models: int
    max_batch_size: int
    max_input_tokens: int
    truncate_long_inputs: bool
    service_version: str


settings = Settings(
    device=os.getenv("EMBEDDINGS_DEVICE", "auto"),
    cuda_memory_fraction=_get_env_optional_float("EMBEDDINGS_CUDA_MEMORY_FRACTION"),
    bitsandbytes=os.getenv("EMBEDDINGS_BITSANDBYTES", "").strip().lower() or None,
    max_loaded_models=_get_env_int("EMBEDDINGS_MAX_LOADED_MODELS", 1),
    max_batch_size=_get_env_int("EMBEDDINGS_MAX_BATCH_SIZE", 16),
    max_input_tokens=_get_env_int("EMBEDDINGS_MAX_INPUT_TOKENS", 4096),
    truncate_long_inputs=_get_env_bool("EMBEDDINGS_TRUNCATE_INPUTS", False),
    service_version=SERVICE_VERSION,
)

if settings.max_loaded_models < 1:
    raise RuntimeError("EMBEDDINGS_MAX_LOADED_MODELS must be >= 1")

if settings.max_batch_size < 1:
    raise RuntimeError("EMBEDDINGS_MAX_BATCH_SIZE must be >= 1")

if settings.max_input_tokens < 1:
    raise RuntimeError("EMBEDDINGS_MAX_INPUT_TOKENS must be >= 1")

if settings.cuda_memory_fraction is not None and not (
    0.0 < settings.cuda_memory_fraction <= 1.0
):
    raise RuntimeError("EMBEDDINGS_CUDA_MEMORY_FRACTION must be > 0.0 and <= 1.0")

if settings.bitsandbytes is not None and settings.bitsandbytes not in {"4bit", "8bit"}:
    raise RuntimeError("EMBEDDINGS_BITSANDBYTES must be '4bit', '8bit', or empty")
