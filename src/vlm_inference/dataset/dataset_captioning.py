import logging
from pathlib import Path
from typing import Type

import hydra
from jinja2 import Template
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from ..utils import parse_pydantic_schema
from .dataset_base import Dataset, ImageExample

logger = logging.getLogger(__name__)


class CaptionResponse(PydanticBaseModel):
    caption: str = Field(description="Caption for the image")


class CulturalCaptionResponse(PydanticBaseModel):
    caption: str = Field(description="Caption for the image")
    is_cultural: bool = Field(description="true/false")
    justification: str = Field(
        description="Why or why not the image contains cultural information"
    )


class ImageCaptioningDataset(Dataset):
    name = "image_captioning"
    json_schema: Type[PydanticBaseModel] = CaptionResponse

    def __init__(self, path: str, template_name: str):
        self._load_dataset(Path(path))
        self._load_template(template_name)

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

    def __getitem__(self, index: int) -> ImageExample:
        image_path = self.data[index]
        prompt = self.get_prompt()

        return ImageExample(image_path=image_path, prompt=prompt)

    def __len__(self) -> int:
        return len(self.data)


class CulturalImageCaptioningDataset(ImageCaptioningDataset):
    name = "cultural_image_captioning"
    json_schema: Type[PydanticBaseModel] = CulturalCaptionResponse
