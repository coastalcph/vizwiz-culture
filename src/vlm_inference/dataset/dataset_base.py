import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel as PydanticBaseModel

logger = logging.getLogger(__name__)


@dataclass
class ImageExample:
    image_path: str
    prompt: str


class Dataset(ABC):
    name: str
    json_schema: Type[PydanticBaseModel]

    @abstractmethod
    def _load_dataset(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def get_prompt(self, *args, **kwargs) -> str: ...

    @abstractmethod
    def __getitem__(self, index: int) -> ImageExample: ...

    @abstractmethod
    def __len__(self) -> int: ...
