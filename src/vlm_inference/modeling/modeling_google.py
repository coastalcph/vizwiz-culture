import logging
import os
from typing import Any, Dict, Optional, Tuple, Type

import vertexai  # type: ignore
from pydantic import BaseModel as PydanticBaseModel
from vertexai.preview.generative_models import (  # type: ignore
    GenerationConfig, GenerativeModel, Part)

from ..configuration import Pricing
from ..dataset import ImageExample
from ..utils import read_image_as_bytes
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)


class GoogleModel(VisionLanguageModel):

    def __init__(
        self, name: str, generation_kwargs: Dict[str, Any], json_mode: bool, pricing: Pricing
    ):
        super().__init__(name, generation_kwargs, json_mode)

        logger.info("Using Google API")

        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))

        self.model = GenerativeModel(self.name)
        self.generation_config = GenerationConfig(
            **self.generation_kwargs,
            response_mime_type=(
                "application/json" if self.json_mode and "1.5" in self.name else "text/plain"
            )
        )
        self.pricing = pricing

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:
        img = Part.from_data(data=read_image_as_bytes(example.image_path), mime_type="image/jpeg")

        response = self.model.generate_content(
            [img, example.prompt], generation_config=self.generation_config
        )
        generated_text = response.text
        usage_metadata = UsageMetadata(
            input_token_count=response.usage_metadata.prompt_token_count,
            output_token_count=response.usage_metadata.candidates_token_count,
        )
        return generated_text, usage_metadata
