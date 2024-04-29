from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    type: str = MISSING
    path: str = MISSING
    template_name: str = MISSING
