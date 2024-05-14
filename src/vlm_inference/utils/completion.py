from dataclasses import dataclass
from typing import Any, Dict, List

from ..dataset import ImageExample


@dataclass
class Completion:
    example: ImageExample
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
