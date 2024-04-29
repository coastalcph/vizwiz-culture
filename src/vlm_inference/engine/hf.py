import logging
from typing import Optional

import torch
from hydra.utils import instantiate
from outlines.integrations.transformers import JSONPrefixAllowedTokens
from PIL import Image
from pydantic import BaseModel
from transformers.feature_extraction_utils import BatchFeature

from ..configuration.models import HfModelConfig
from ..dataset.base import Example
from ..utils.misc import as_dict, torch_dtype_from_str
from .base import BaseEngine

logger = logging.getLogger(__name__)


class HfEngine(BaseEngine):

    def __init__(self, config: HfModelConfig):
        super().__init__(config)

        self.model = instantiate(
            config.model_cls,
            pretrained_model_name_or_path=config.name,
            torch_dtype=torch_dtype_from_str(config.dtype),
        )

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = instantiate(
            config.processor_cls, pretrained_model_name_or_path=config.name
        )

        self.generation_kwargs = as_dict(config.generation_kwargs)
        self.gen_max_retries = self.generation_kwargs.pop("max_retries", 10)

    def _extract_features(self, example: Example) -> BatchFeature:
        return self.processor(
            images=Image.open(example.image_path), text=example.prompt, return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)

    def _generate(self, example: Example, json_schema: Optional[BaseModel] = None) -> str:

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
        if self.config.strip_prompt:
            generated_tokens = generated_tokens[:, features.input_ids.shape[1] :]

        generated_tokens = generated_tokens.cpu()

        generated_text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[
            0
        ].strip()

        return generated_text
