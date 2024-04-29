import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel
from tqdm import tqdm

from ..configuration.models import ApiType, ModelConfig
from ..dataset.base import BaseDataset, Example
from ..utils.json import parse_json, validate_json_with_schema
from ..utils.misc import as_dict

logger = logging.getLogger(__name__)


@dataclass
class Completion:
    example: Example
    response: Any

    def values_as_list(self) -> List[Any]:
        example_values = list(self.example.__dict__.values())
        if isinstance(self.response, dict):
            response_values = list(self.response.values())
        else:
            response_values = [self.response]
        return example_values + response_values

    def keys_as_list(self) -> List[str]:
        example_keys = list(self.example.__dict__.keys())
        if isinstance(self.response, dict):
            response_keys = list(self.response.keys())
        else:
            response_keys = ["response"]
        return example_keys + response_keys


@dataclass
class StringCompletion(Completion):
    response: str


@dataclass
class JsonCompletion(Completion):
    response: Dict[str, Any]


class BaseEngine(ABC):
    """Base class for all engines."""

    def __init__(self, config: ModelConfig):
        self.config = config

        self.generation_kwargs = as_dict(config.generation_kwargs)
        self.gen_max_retries = self.generation_kwargs.pop("max_retries", 50)
        self.gen_sleep_duration = self.generation_kwargs.pop("sleep_duration", 0)

    @abstractmethod
    def _generate(self, example: Example, json_schema: Optional[BaseModel] = None) -> str: ...

    def json_step(self, example: Example, json_schema: BaseModel) -> JsonCompletion:
        for retry in range(self.gen_max_retries):
            try:
                generated_json = parse_json(self._generate(example, json_schema))
            except Exception as e:
                logger.warning(e)
                generated_json = None
                time.sleep(self.gen_sleep_duration)

            if generated_json is not None and validate_json_with_schema(
                generated_json, json_schema
            ):
                break

        if generated_json is None:
            logger.warning(f"Failed to generate JSON after {self.gen_max_retries} iterations")
            return JsonCompletion(
                example=example,
                response={field: None for field in json_schema.model_fields.keys()},
            )

        logger.info(f"Generated JSON after {retry + 1} iterations")

        return JsonCompletion(example=example, response=generated_json)

    def step(self, example: Example) -> StringCompletion:
        for retry in range(self.gen_max_retries):
            try:
                generated_text = self._generate(example)
            except Exception as e:
                logger.warning(e)
                generated_text = ""
                time.sleep(self.gen_sleep_duration)

            if generated_text.strip():
                break

        if not generated_text:
            logger.warning(f"Failed to generate text after {self.gen_max_retries} iterations")
            return StringCompletion(example=example, response="")

        logger.info(f"Generated JSON after {retry + 1} iterations")

        return StringCompletion(example=example, response=generated_text)

    def run(self, dataset: BaseDataset, callbacks: List[Callable] = []):
        step_cls = (
            partial(self.json_step, json_schema=dataset.json_schema)
            if self.config.json_mode
            else self.step
        )

        for i in tqdm(range(len(dataset))):
            completion = step_cls(dataset[i])
            for callback in callbacks:
                callback(i, completion)


def create_engine(config: ModelConfig) -> BaseEngine:
    """Create an engine from the given configuration."""
    if config.api_type == ApiType.HF:
        from .hf import HfEngine

        return HfEngine(config)
    elif config.api_type == ApiType.OPENAI:
        from .openai import OpenaiEngine

        return OpenaiEngine(config)
    elif config.api_type == ApiType.GOOGLE:
        from .google import GoogleEngine

        return GoogleEngine(config)
    else:
        raise ValueError(f"Unsupported API type: {config.api_type}")
