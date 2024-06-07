import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel

from ..dataset.dataset_base import ImageExample
from ..utils.misc import torch_dtype_from_str
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)


class CpmModel(VisionLanguageModel):

    def __init__(
        self,
        name: str,
        generation_kwargs: Dict[str, Any],
        json_mode: bool,
        dtype: str,
        model_cls: Callable,
        processor_cls: Callable,
        **kwargs
    ):
        super().__init__(name, generation_kwargs, json_mode)

        self.model = model_cls(
            pretrained_model_name_or_path=self.name,
            torch_dtype=torch_dtype_from_str(dtype),
        )

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = processor_cls(pretrained_model_name_or_path=self.name)

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:

        generated_text = self.model.chat(
            image=Image.open(example.image_path).convert("RGB"),
            msgs=[{"role": "user", "content": example.prompt}],
            tokenizer=self.processor,
            **self.generation_kwargs
        ).strip()

        usage_metadata = UsageMetadata(
            input_token_count=0,
            output_token_count=0,
        )

        return generated_text, usage_metadata
