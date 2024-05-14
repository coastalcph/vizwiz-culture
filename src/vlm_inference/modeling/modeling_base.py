import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel as PydanticBaseModel

from ..dataset import ImageExample
from ..utils.usage_tracking import UsageMetadata

logger = logging.getLogger(__name__)


class VisionLanguageModel(ABC):
    """Base class for all models."""

    def __init__(self, name: str, generation_kwargs: Dict[str, Any], json_mode: bool):

        self.name = name
        self.generation_kwargs = generation_kwargs
        self.gen_max_retries = self.generation_kwargs.pop("max_retries", 50)
        self.gen_sleep_duration = self.generation_kwargs.pop("sleep_duration", 0)
        self.json_mode = json_mode

    @abstractmethod
    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]: ...
