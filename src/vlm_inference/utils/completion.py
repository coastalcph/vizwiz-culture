from dataclasses import dataclass
from typing import Any, Dict, List, Union

from ..dataset.dataset_base import ImageExample

StringResponse = str
JsonResponse = Dict[str, Any]

@dataclass
class Completion:
    example: ImageExample
    response: Union[StringResponse, JsonResponse]

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
    response: StringResponse


@dataclass
class JsonCompletion(Completion):
    response: JsonResponse
