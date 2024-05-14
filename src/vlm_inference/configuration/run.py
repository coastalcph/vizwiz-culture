from dataclasses import dataclass
from typing import Any, Dict, List

from omegaconf import MISSING

from .callbacks import CallbackConfig
from .dataset import DatasetConfig
from .models import ModelConfig


@dataclass
class BaseConfig:
    _callback_dict: Dict[str, Any] = MISSING


@dataclass
class RunConfig(BaseConfig):
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    callbacks: List[CallbackConfig] = MISSING
