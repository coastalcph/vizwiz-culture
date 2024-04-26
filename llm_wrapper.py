import base64
import datetime
import io
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import openai
import pandas as pd
import torch
import vertexai
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoProcessor,
                          Blip2ForConditionalGeneration, Blip2Processor,
                          InstructBlipForConditionalGeneration,
                          InstructBlipProcessor)
from vertexai.preview.generative_models import (GenerationConfig,
                                                GenerativeModel, Part)

from dataset import Dataset, Example

logger = logging.getLogger(__name__)


class SaveStrategy(Enum):
    LOCAL = "local"
    WANDB = "wandb"


@dataclass
class Completion:
    prompt: str
    response: Union[str, Dict[str, Any]]


@dataclass
class ModelOutput:
    model_id: str
    example_data: Example
    completions: List[Completion]

    def as_nested_list(self) -> List[List[str]]:
        outer = []
        for c in self.completions:
            inner = [self.model_id] + list(self.example_data.__dict__.values()) + [c.prompt]
            if isinstance(c.response, dict):
                inner.extend(list(c.response.values()))
            else:
                inner.append(c.response)

            outer.append(inner)
        return outer


def parse_response_as_json(response: str, expected_keys: List[str], fallback_fn: Callable):
    # Try to force response into JSON format
    response = response.replace("```", "").strip()
    if "}" in response:
        response = response.split("}")[0] + "}"
    if not response.endswith("}"):
        response += "}"
    try:
        parsed_response = json.loads(response)
        if not all(k in parsed_response for k in expected_keys):
            raise RuntimeError("Missing keys in parsed response")
    except Exception:
        # Fallback to custom parsing function if json.loads fails
        parsed_response = fallback_fn(response)
    logger.info(f"{json.dumps(parsed_response, ensure_ascii=False, indent=4)}\n{'-' * 100}")
    return parsed_response


class Wrapper(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        self.model_config = MODEL_CONFIGS[config.model_name]
        self.resolved_model_id = self.model_config.model_id
        self.curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @abstractmethod
    def inference_step(
        self,
        example: Dict[str, Union[Example, List[str]]],
        json_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        ...

    def get_completion(
        self, prompt: str, response: str, json_kwargs: Optional[Dict[str, Any]] = None
    ) -> Completion:
        logger.info(f"{'-' * 100}\nPrompt: {prompt}")
        if self.config.parse_json:
            assert json_kwargs is not None
            response = parse_response_as_json(
                response, json_kwargs["expected_keys"], json_kwargs["fallback_fn"]
            )
        else:
            logger.info(f"{response}\n{'-' * 100}")
        return Completion(prompt, response)

    def save_outputs(self, outputs: List[ModelOutput], dataset_name: str) -> pd.DataFrame:
        # Flatten
        all_outputs = []
        for output in outputs:
            all_outputs.extend(output.as_nested_list())

        columns = ["model_id"] + list(outputs[0].example_data.__dict__.keys()) + ["prompt"]
        if self.config.parse_json:
            columns.extend(list(outputs[0].completions[0].response.keys()))
        else:
            columns.append("response")

        # Convert to dataframe
        df = pd.DataFrame(all_outputs, columns=columns)

        # Save
        if self.config.save_strategy == SaveStrategy.LOCAL:
            Path.mkdir(Path(self.config.save_path), parents=True, exist_ok=True)
            output_filepath = (
                Path(self.config.save_path)
                / f"outputs_{os.path.basename(self.resolved_model_id)}_{dataset_name}_{self.config.template_name}_{self.curr_time}.csv"
            )
            df.to_csv(output_filepath, sep="\t", index=False)

            logger.info(f"Saved outputs to `{output_filepath}`")

        elif self.config.save_strategy == SaveStrategy.WANDB:
            wandb.init(
                project=dataset_name,
                name=f"{os.path.basename(self.resolved_model_id)}_{dataset_name}_{self.config.template_name}_{self.curr_time}",
                id=f"{os.path.basename(self.resolved_model_id)}_{dataset_name}_{self.config.template_name}_{self.curr_time}",
                config=OmegaConf.to_container(self.config),
            )
            wandb.log({"Result": wandb.Table(dataframe=df)})
            wandb.finish()

            logger.info(f"Saved outputs to wandb")

        return df

    def run_inference(self, ds: Dataset) -> pd.DataFrame:
        json_kwargs = (
            {
                "expected_keys": ds.json_expected_keys,
                "fallback_fn": ds.json_parsing_fallback_fn,
            }
            if self.config.parse_json
            else None
        )

        model_outputs = [
            self.inference_step(ds[i], json_kwargs=json_kwargs) for i in tqdm(range(len(ds)))
        ]
        return self.save_outputs(model_outputs, dataset_name=ds.name)


class OpenaiWrapper(Wrapper):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        logger.info("Using OpenAI API")

        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def inference_step(
        self,
        example: Dict[str, Union[Example, List[str]]],
        json_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        completions = []

        buffered = io.BytesIO()
        example["img"].save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        for prompt_text in example["prompts"]:
            while True:
                try:
                    response = self.client.chat.completions.create(
                        model=self.resolved_model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {
                                        "type": "image",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "auto",
                                        },
                                    },
                                ],
                            }
                        ],
                        **self.model_config.generation_kwargs,
                    )
                    break
                except openai.OpenAIError as e:
                    logger.info(f"Error: {e}")
                    raise e

            generated_text = response.choices[0].message.content

            completions.append(
                self.get_completion(prompt_text, generated_text, json_kwargs=json_kwargs)
            )

        model_output = ModelOutput(
            model_id=self.resolved_model_id,
            example_data=example["data"],
            completions=completions,
        )
        return model_output


class GoogleWrapper(Wrapper):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        logger.info("Using Google API")

        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))

        self.model = GenerativeModel(self.resolved_model_id)
        self.generation_config = GenerationConfig(**self.model_config.generation_kwargs)

    def inference_step(
        self,
        example: Dict[str, Union[Example, Image.Image, List[str]]],
        json_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        completions = []

        buffered = io.BytesIO()
        example["img"].save(buffered, format="JPEG")
        img = Part.from_data(data=buffered.getvalue(), mime_type="image/jpeg")

        for prompt_text in example["prompts"]:
            max_retries = 50
            sleep_duration = 2
            response = None
            for retry in range(max_retries):
                try:
                    response = self.model.generate_content(
                        [img, prompt_text], generation_config=self.generation_config
                    )
                    generated_text = response.text
                    break
                except Exception as e:
                    if (retry + 1) == max_retries:
                        raise RuntimeError(
                            f"Request could not be completed after {max_retries} retries. Error: {e}"
                        )

                    logger.info(f"Request failed with {e}. Retrying in {sleep_duration} seconds")
                    time.sleep(sleep_duration)

            completions.append(
                self.get_completion(prompt_text, generated_text, json_kwargs=json_kwargs)
            )

        model_output = ModelOutput(
            model_id=self.resolved_model_id,
            example_data=example["data"],
            completions=completions,
        )
        return model_output


class HfWrapper(Wrapper):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.model_size = self.config.model_size or self.model_config.default_model_size
        self.resolved_model_id = self.model_config.model_id.format(model_size=self.model_size)

        logger.info(f"Using HuggingFace with local model: {self.resolved_model_id}")

        self.model = self.model_config.model_cls(
            pretrained_model_name_or_path=self.resolved_model_id,
            torch_dtype=self.model_config.dtype,
        )
        self.processor = self.model_config.processor_cls(
            pretrained_model_name_or_path=self.resolved_model_id
        )

    def inference_step(
        self,
        example: Dict[str, Union[Example, Image.Image, List[str]]],
        json_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        logger.info(f"Prompting model {self.resolved_model_id}")
        prompts = [
            self.processor(images=example["img"], text=p, return_tensors="pt").to(
                self.model.device, dtype=self.model.dtype
            )
            for p in example["prompts"]
        ]

        completions = []
        for prompt in prompts:
            generated_text = ""
            iteration = 0
            while not generated_text.strip():
                if iteration % 5 == 0:
                    logger.info(f"Retrying prompt after {iteration} iterations")

                # Generate
                generated_tokens = self.model.generate(
                    **prompt,
                    **self.model_config.generation_kwargs,
                )
                generated_tokens = generated_tokens.cpu()

                # Decode
                prompt_text = self.processor.batch_decode(
                    prompt.input_ids, skip_special_tokens=True
                )[0]
                generated_text = self.processor.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0].strip()

                iteration += 1

            completions.append(
                self.get_completion(prompt_text, generated_text, json_kwargs=json_kwargs)
            )

        model_output = ModelOutput(
            model_id=self.resolved_model_id,
            example_data=example["data"],
            completions=completions,
        )
        return model_output


class ApiType(Enum):
    OPENAI = "openai"
    HF = "hf"
    GOOGLE = "google"


@dataclass
class OpenaiApiConfig:
    model_id: str = "gpt-4-vision-preview"
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"max_tokens": 300, "temperature": 1.0}
    )
    api_type: ApiType = ApiType.OPENAI
    wrapper_cls: Wrapper = OpenaiWrapper


@dataclass
class GoogleApiConfig:
    model_id: str = "gemini-1.0-pro-vision"
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.4,  # this is the API default value
            "max_output_tokens": 300,
        }
    )
    api_type: ApiType = ApiType.GOOGLE
    wrapper_cls: Wrapper = GoogleWrapper


@dataclass
class HFApiConfig:
    model_id: str
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 300,
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
        }
    )
    dtype: torch.dtype = torch.float16
    default_model_size: str = "2.7b"
    model_cls: Callable = partial(AutoModelForCausalLM.from_pretrained, device_map="auto")
    processor_cls: Callable = partial(AutoProcessor.from_pretrained, use_fast=True)
    api_type: ApiType = ApiType.HF
    wrapper_cls: Wrapper = HfWrapper


MODEL_CONFIGS = {
    "gpt-4": OpenaiApiConfig(),
    "gemini": GoogleApiConfig(),
    "blip2": HFApiConfig(
        model_id="Salesforce/blip2-opt-{model_size}",
        model_cls=partial(
            Blip2ForConditionalGeneration.from_pretrained,
            device_map="auto",
        ),
        processor_cls=partial(Blip2Processor.from_pretrained, use_fast=True),
        default_model_size="2.7b",
    ),
    "instructblip": HFApiConfig(
        model_id="Salesforce/instructblip-vicuna-{model_size}",
        model_cls=partial(
            InstructBlipForConditionalGeneration.from_pretrained,
            device_map="auto",
        ),
        processor_cls=partial(InstructBlipProcessor.from_pretrained, use_fast=True),
        default_model_size="7b",
    ),
}


def get_llm_wrapper(config: DictConfig) -> Wrapper:
    if config.model_name not in MODEL_CONFIGS:
        raise RuntimeError(f"Model {config.model_name} is not supported.")

    return MODEL_CONFIGS[config.model_name].wrapper_cls(config)
