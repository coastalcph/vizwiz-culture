import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from jinja2 import Template
from omegaconf import DictConfig
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Example: ...


@dataclass
class ImageCaptioningExample(Example):
    image_path: str


class Dataset(ABC):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def _load_dataset(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def get_prompts(self, *args, **kwargs) -> List[str]: ...

    @abstractmethod
    def json_parsing_fallback_fn(self, *args, **kwargs) -> Dict[str, Any]: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]: ...

    def __len__(self) -> int:
        return len(self.data)


class ImageCaptioningDataset(Dataset):
    json_expected_keys = ["caption"]
    name = "image_captioning"

    def __init__(self, data_dir: Path, template_filepath: Path):
        self._load_dataset(data_dir)
        self._load_template(template_filepath)

    def _load_dataset(self, data_dir: Path) -> None:
        self.data = [
            ImageCaptioningExample(str(image_path)) for image_path in data_dir.rglob("*.jpg")
        ]
        logger.info(f"Resolved {len(self.data)} images in directory `{data_dir}`.")

    def _load_template(self, template_path: Path) -> None:
        with open(template_path) as f:
            self.template: Template = Template(f.read())

    def get_prompts(self, **kwargs) -> List[str]:
        return [self.template.render(**kwargs)]

    def json_parsing_fallback_fn(self, response: str) -> Dict[str, str]:
        return {"caption": response}

    def __getitem__(self, index: int) -> Dict[str, Union[ImageCaptioningExample, List[str]]]:
        example = self.data[index]
        img = Image.open(example.image_path)
        prompts = self.get_prompts(**example.__dict__)

        return {"data": example, "img": img, "prompts": prompts}


class CulturalImageCaptioningDataset(ImageCaptioningDataset):
    json_expected_keys = ["cultural_information", "justification", "caption"]
    name = "cultural_image_captioning"

    def json_parsing_fallback_fn(self, response: str) -> Dict[str, str]:
        return {"cultural_information": False, "justification": "", "caption": response}

def get_dataset(config: DictConfig) -> Dataset:
    if config.dataset_type == "captioning":
        dataset_cls = ImageCaptioningDataset
    if config.dataset_type == "cultural_captioning":
        dataset_cls = CulturalImageCaptioningDataset
    else:
        raise NotImplementedError(f"Dataset type {config.dataset_type} not implemented.")

    return dataset_cls(
        Path(config.dataset_path), Path("templates") / f"{config.template_name}.txt"
    )
