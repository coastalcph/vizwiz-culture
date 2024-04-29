import logging
import os
from typing import Optional

import anthropic
from pydantic import BaseModel

from ..configuration.models import AnthropicModelConfig
from ..dataset.base import Example
from ..utils.misc import read_image_as_b64
from .base import BaseEngine

logger = logging.getLogger(__name__)


class AnthropicEngine(BaseEngine):

    def __init__(
        self,
        config: AnthropicModelConfig,
    ):
        super().__init__(config)

        logger.info("Using Anthropic API")

        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    def _generate(self, example: Example, json_schema: Optional[BaseModel] = None) -> str:
        base64_image = read_image_as_b64(example.image_path)

        response = self.client.messages.create(
            model=self.config.name,
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
        return response.content[0].text
