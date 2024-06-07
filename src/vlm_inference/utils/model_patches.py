from typing import List, Optional, Union

from transformers import AutoTokenizer, BatchEncoding, ProcessorMixin, TensorType
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import PaddingStrategy


class ChatGLMProcessor(ProcessorMixin):
    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ],
        images: Optional[ImageInput] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        do_pad: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchEncoding:

        if isinstance(images, list):
            raise ValueError("ChatGLM currently does not support multiple images")
        if isinstance(text, list):
            raise ValueError("ChatGLM currently does not support multiple texts")

        return self.tokenizer.apply_chat_template(
            [{"role": "user", "image": images, "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=return_tensors,
            return_dict=True,
        )
