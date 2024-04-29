import logging
import os
from typing import Optional

import openai
from pydantic import BaseModel

from ..configuration.models import OpenaiModelConfig
from ..dataset.base import Example
from ..utils.misc import read_image_as_b64
from .base import BaseEngine

logger = logging.getLogger(__name__)


class OpenaiEngine(BaseEngine):

    def __init__(
        self,
        config: OpenaiModelConfig,
    ):
        super().__init__(config)

        logger.info("Using OpenAI API")

        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def _generate(self, example: Example, json_schema: Optional[BaseModel] = None) -> str:
        base64_image = read_image_as_b64(example.image_path)
        response = self.client.chat.completions.create(
            model=self.config.name,
            response_format={
                "type": (
                    "json_object"
                    if json_schema is not None and "preview" not in self.config.name
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
        return response.choices[0].message.content
