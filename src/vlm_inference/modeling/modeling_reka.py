import logging
import os
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel as PydanticBaseModel
from reka import ChatMessage
from reka.client import Reka

from ..configuration.models import Pricing
from ..dataset.dataset_base import ImageExample
from ..utils.misc import read_image_as_b64
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)


class RekaModel(VisionLanguageModel):

    def __init__(
        self, name: str, generation_kwargs: Dict[str, Any], json_mode: bool, pricing: Pricing
    ):
        super().__init__(name, generation_kwargs, json_mode)

        logger.info("Using Reka API")

        self.client = Reka(
            api_key=os.environ.get("REKA_API_KEY"),
        )
        self.pricing = pricing

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:
        base64_image = read_image_as_b64(example.image_path)

        response = self.client.chat.create(
            messages=[
                ChatMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                        {"type": "text", "text": example.prompt},
                    ],
                    role="user",
                )
            ],
            model=self.name,
        )

        generated_text = response.responses[0].message.content
        usage_metadata = UsageMetadata(
            input_token_count=response.usage.input_tokens,
            output_token_count=response.usage.output_tokens,
        )
        return generated_text, usage_metadata
