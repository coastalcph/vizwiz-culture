import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
from jinja2 import Template
from pydantic import BaseModel, Field

from ..configuration.dataset import DatasetConfig
from ..utils.json import parse_pydantic_schema
from .base import BaseDataset, Example

logger = logging.getLogger(__name__)


@dataclass
class ImageCaptioningExample(Example):
    image_path: str
    prompt: str


class CaptionResponse(BaseModel):
    caption: str = Field(description="Caption for the image")


class CulturalCaptionResponse(BaseModel):
    caption: str = Field(description="Caption for the image")
    is_cultural: bool = Field(description="true/false")
    justification: str = Field(
        description="Why or why not the image contains cultural information"
    )


class ImageCaptioningDataset(BaseDataset):
    name = "image_captioning"
    json_schema = CaptionResponse

    def __init__(self, config: DatasetConfig):
        self._load_dataset(Path(config.path))
        self._load_template(config.template_name)

    def _load_dataset(self, data_dir: Path) -> None:
        if not data_dir.exists() and not data_dir.is_dir():
            raise FileNotFoundError(f"Directory `{data_dir}` does not exist.")

        self.data = [str(image_path.resolve()) for image_path in data_dir.rglob("*.jpg")]
        logger.info(f"Resolved {len(self.data)} images in directory `{data_dir}`.")

    def _load_template(self, template_name: str) -> None:
        template_path = Path(hydra.utils.get_original_cwd()) / "templates" / f"{template_name}.txt"
        with open(template_path) as f:
            self.template: Template = Template(f.read())

    def get_prompt(self) -> str:
        return self.template.render(json_schema=parse_pydantic_schema(self.json_schema))

    def __getitem__(self, index: int) -> ImageCaptioningExample:
        image_path = self.data[index]
        prompt = self.get_prompt()

        return ImageCaptioningExample(image_path=image_path, prompt=prompt)


class CulturalImageCaptioningDataset(ImageCaptioningDataset):
    name = "cultural_image_captioning"
    json_schema = CulturalCaptionResponse
