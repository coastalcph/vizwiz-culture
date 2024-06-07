import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Type

import hydra
from jinja2 import Template
from pydantic import BaseModel as PydanticBaseModel

from ..utils.json_parsing import parse_pydantic_schema

logger = logging.getLogger(__name__)


@dataclass
class ImageExample:
    image_path: str
    prompt: str


class ImageDataset:
    name: str
    json_schema: Type[PydanticBaseModel]

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

    @cache
    def get_prompt(self) -> str:
        return self.template.render(json_schema=parse_pydantic_schema(self.json_schema))

    def __getitem__(self, index: int) -> ImageExample:
        image_path = self.data[index]
        prompt = self.get_prompt()

        return ImageExample(image_path=image_path, prompt=prompt)

    def __len__(self) -> int:
        return len(self.data)
