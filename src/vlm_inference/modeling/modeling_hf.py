import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from outlines.integrations.transformers import JSONPrefixAllowedTokens
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel
from transformers.feature_extraction_utils import BatchFeature

from ..dataset import ImageExample
from ..utils import torch_dtype_from_str
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)


class HfModel(VisionLanguageModel):

    def __init__(
        self,
        name: str,
        generation_kwargs: Dict[str, Any],
        json_mode: bool,
        dtype: str,
        model_cls: Callable,
        processor_cls: Callable,
        strip_prompt: bool = False,
        **kwargs
    ):
        super().__init__(name, generation_kwargs, json_mode)

        self.model = model_cls(
            pretrained_model_name_or_path=self.name,
            torch_dtype=torch_dtype_from_str(dtype),
        )

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = processor_cls(pretrained_model_name_or_path=self.name)
        self.strip_prompt = strip_prompt

    def _extract_features(self, example: ImageExample) -> BatchFeature:
        return self.processor(
            images=Image.open(example.image_path), text=example.prompt, return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:

        features = self._extract_features(example)

        prefix_allowed_tokens_fn = (
            JSONPrefixAllowedTokens(
                schema=json_schema,
                tokenizer_or_pipe=self.processor.tokenizer,
                whitespace_pattern=r" ?",
            )
            if json_schema is not None
            else None
        )

        generated_tokens = self.model.generate(
            **features,
            **self.generation_kwargs,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        if self.strip_prompt:
            generated_tokens = generated_tokens[:, features.input_ids.shape[1] :]

        generated_tokens = generated_tokens.cpu()

        generated_text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[
            0
        ].strip()

        usage_metadata = UsageMetadata(
            input_token_count=features.input_ids.shape[1],
            output_token_count=generated_tokens.shape[1],
        )

        return generated_text, usage_metadata
