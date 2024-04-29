from dataclasses import dataclass
from typing import Any, Dict

from omegaconf import MISSING


@dataclass
class BaseConfig:
    _callback_dict: Dict[str, Any] = MISSING
