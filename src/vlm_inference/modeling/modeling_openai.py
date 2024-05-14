import logging
import os
from typing import Any, Dict, Optional, Tuple, Type

from openai import OpenAI
from pydantic import BaseModel as PydanticBaseModel

from ..configuration import Pricing
from ..dataset import ImageExample
from ..utils import read_image_as_b64
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)


class OpenaiModel(VisionLanguageModel):

    def __init__(
        self, name: str, generation_kwargs: Dict[str, Any], json_mode: bool, pricing: Pricing
    ):
        super().__init__(name, generation_kwargs, json_mode)

        logger.info("Using OpenAI API")

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.pricing = pricing

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:
        base64_image = read_image_as_b64(example.image_path)
        response = self.client.chat.completions.create(
            model=self.name,
            response_format={
                "type": (
                    "json_object"
                    if json_schema is not None and "preview" not in self.name
                    else "text"
                )
            },  # the vision preview doesn't support JSON mode
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            **self.generation_kwargs,
        )
        generated_text = response.choices[0].message.content
        usage_metadata = UsageMetadata(
            input_token_count=response.usage.prompt_tokens,
            output_token_count=response.usage.completion_tokens,
        )
        return generated_text, usage_metadata
