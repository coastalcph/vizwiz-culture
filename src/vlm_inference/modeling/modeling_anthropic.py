import logging
import os
from typing import Any, Dict, Optional, Tuple, Type

from anthropic import Anthropic
from pydantic import BaseModel as PydanticBaseModel

from ..configuration.models import Pricing
from ..dataset.dataset_base import ImageExample
from ..utils.misc import read_image_as_b64
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)


class AnthropicModel(VisionLanguageModel):

    def __init__(
        self, name: str, generation_kwargs: Dict[str, Any], json_mode: bool, pricing: Pricing
    ):
        super().__init__(name, generation_kwargs, json_mode)

        logger.info("Using Anthropic API")

        self.client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.pricing = pricing

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:
        base64_image = read_image_as_b64(example.image_path)

        response = self.client.messages.create(
            model=self.name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": example.prompt,
                        },
                    ],
                }
            ],
            **self.generation_kwargs,
        )

        generated_text = response.content[0].text
        usage_metadata = UsageMetadata(
            input_token_count=response.usage.input_tokens,
            output_token_count=response.usage.output_tokens,
        )
        return generated_text, usage_metadata
