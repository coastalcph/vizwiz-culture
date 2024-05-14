from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    _target_: str = MISSING
    path: str = MISSING
    template_name: str = MISSING
