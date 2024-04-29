import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from ..configuration.dataset import DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class Example: ...


class BaseDataset(ABC):
    name: str = "base"
    json_schema: BaseModel = None

    @abstractmethod
    def _load_dataset(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def get_prompt(self, *args, **kwargs) -> str: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Example: ...

    def __len__(self) -> int:
        return len(self.data)


def create_dataset(config: DatasetConfig) -> BaseDataset:
    if config.type == "captioning":
        from .captioning import ImageCaptioningDataset

        return ImageCaptioningDataset(config)
    elif config.type == "cultural_captioning":
        from .captioning import CulturalImageCaptioningDataset

        return CulturalImageCaptioningDataset(config)
    else:
        raise ValueError(f"Unsupported dataset type `{config.type}`")
