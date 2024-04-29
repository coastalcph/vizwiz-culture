import logging
import os
from typing import Optional

import vertexai
from pydantic import BaseModel
from vertexai.preview.generative_models import (GenerationConfig,
                                                GenerativeModel, Part)

from ..configuration.models import GoogleModelConfig
from ..dataset.base import Example
from ..utils.misc import read_image_as_bytes
from .base import BaseEngine

logger = logging.getLogger(__name__)


class GoogleEngine(BaseEngine):

    def __init__(
        self,
        config: GoogleModelConfig,
    ):
        super().__init__(config)

        logger.info("Using Google API")

        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))

        self.model = GenerativeModel(config.name)
        self.generation_config = GenerationConfig(
            **self.generation_kwargs,
            # FIXME: This is not yet available in the API
            # response_mime_type=(
            #     "application/json"
            #     if self.config.json_mode and "1.5" in config.name
            #     else "text/plain"
            # )
        )

    def _generate(self, example: Example, json_schema: Optional[BaseModel] = None) -> str:
        img = Part.from_data(data=read_image_as_bytes(example.image_path), mime_type="image/jpeg")

        response = self.model.generate_content(
            [img, example.prompt], generation_config=self.generation_config
        )
        return response.text
