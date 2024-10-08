from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class CallbackConfig:
    _target_: str = MISSING
    _partial_: bool = True


@dataclass
class SaveToCsvCallbackConfig(CallbackConfig):
    _target_: str = "vlm_inference.SaveToCsvCallback"
    file_path: str = "output.csv"


@dataclass
class LoggingCallbackConfig(CallbackConfig):
    _target_: str = "vlm_inference.LoggingCallback"


@dataclass
class WandbCallbackConfig(CallbackConfig):
    _target_: str = "vlm_inference.WandbCallback"
    project: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    table_name: str = "results"
    log_every: int = 50


@dataclass
class CostLoggingCallbackConfig(CallbackConfig):
    _target_: str = "vlm_inference.CostLoggingCallback"
    log_every: int = 50
