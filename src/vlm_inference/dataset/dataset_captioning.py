from typing import Type

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from .dataset_base import ImageDataset


class CaptionResponse(PydanticBaseModel):
    caption: str = Field(description="Caption for the image")


class ImageCaptioningDataset(ImageDataset):
    name = "image_captioning"
    json_schema: Type[PydanticBaseModel] = CaptionResponse


class CulturalCaptionResponse(PydanticBaseModel):
    caption: str = Field(description="Caption for the image")
    is_cultural: bool = Field(description="true/false")
    justification: str = Field(
        description="Why or why not the image contains cultural information"
    )


class CulturalImageCaptioningDataset(ImageCaptioningDataset):
    name = "cultural_image_captioning"
    json_schema: Type[PydanticBaseModel] = CulturalCaptionResponse
