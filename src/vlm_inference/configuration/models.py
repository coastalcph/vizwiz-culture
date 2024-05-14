from dataclasses import dataclass, field
from typing import Any, Dict

from omegaconf import MISSING


@dataclass
class Pricing:
    usd_per_input_unit: str
    usd_per_output_unit: str
    unit_tokens: str = "1000000"


@dataclass
class ModelConfig:
    _target_: str = MISSING
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = MISSING
    json_mode: bool = False


@dataclass
class GoogleModelConfig(ModelConfig):
    _target_: str = "vlm_inference.GoogleModel"
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_output_tokens": 300,
            "temperature": 0.5,
            "max_retries": 50,
            "sleep_duration": 2,
        }
    )
    pricing: Pricing = MISSING


@dataclass
class OpenaiModelConfig(ModelConfig):
    _target_: str = "vlm_inference.OpenaiModel"
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 300,
            "temperature": 0.5,
            "max_retries": 50,
            "sleep_duration": 2,
        }
    )
    pricing: Pricing = MISSING


@dataclass
class AnthropicModelConfig(ModelConfig):
    _target_: str = "vlm_inference.AnthropicModel"
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 300,
            "temperature": 0.5,
            "max_retries": 50,
            "sleep_duration": 2,
        }
    )
    pricing: Pricing = MISSING


@dataclass
class HfModel:
    _target_: str = MISSING
    _partial_: bool = True
    low_cpu_mem_usage: bool = True
    attn_implementation: str = "eager"


@dataclass
class HfProcessor:
    _target_: str = MISSING
    _partial_: bool = True
    use_fast: bool = False


@dataclass
class HfModelConfig(ModelConfig):
    _target_: str = "vlm_inference.HfModel"
    name: str = MISSING
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_return_sequences": 1,
            "do_sample": True,
            "max_new_tokens": 300,
            "temperature": 0.5,
            "top_k": 50,
            "max_retries": 10,
        }
    )
    size: str = MISSING
    dtype: str = MISSING
    model_cls: HfModel = MISSING
    processor_cls: HfProcessor = MISSING
    strip_prompt: bool = False
