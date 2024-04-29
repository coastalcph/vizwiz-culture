from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from omegaconf import MISSING


class ApiType(Enum):
    HF = "hf"
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    api_type: ApiType = MISSING
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = MISSING
    json_mode: bool = False


@dataclass
class GoogleModelConfig(ModelConfig):
    api_type: ApiType = ApiType.GOOGLE
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_output_tokens": 300,
            "temperature": 0.4,
            "max_retries": 50,
            "sleep_duration": 2,
        }
    )


@dataclass
class OpenaiModelConfig(ModelConfig):
    api_type: ApiType = ApiType.OPENAI
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 300,
            "temperature": 1.0,
            "max_retries": 50,
            "sleep_duration": 2,
        }
    )


@dataclass
class AnthropicModelConfig(ModelConfig):
    api_type: ApiType = ApiType.ANTHROPIC
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 300,
            "temperature": 1.0,
            "max_retries": 50,
            "sleep_duration": 2,
        }
    )


@dataclass
class HfModel:
    _target_: str = MISSING
    low_cpu_mem_usage: bool = True
    attn_implementation: str = "eager"


@dataclass
class HfProcessor:
    _target_: str = MISSING
    use_fast: bool = False


@dataclass
class HfModelConfig(ModelConfig):
    api_type: ApiType = ApiType.HF
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_return_sequences": 1,
            "do_sample": True,
            "max_new_tokens": 300,
            "temperature": 1.0,
            "top_k": 50,
            "max_retries": 10,
        }
    )
    size: str = MISSING
    dtype: str = MISSING
    model_cls: HfModel = MISSING
    processor_cls: HfProcessor = MISSING
    strip_prompt: bool = False
