from hydra.core.config_store import ConfigStore
from omegaconf import II

from ..utils.misc import is_flashattn_2_supported
from .callbacks import LoggingCallbackConfig, SaveToCsvCallbackConfig, WandbCallbackConfig
from .dataset import DatasetConfig
from .models import (
    AnthropicModelConfig,
    ApiType,
    GoogleModelConfig,
    HfModel,
    HfModelConfig,
    HfProcessor,
    ModelConfig,
    OpenaiModelConfig,
)

cs = ConfigStore.instance()

# Datasets
cs.store(group="dataset", name="captioning", node=DatasetConfig(type="captioning"))
cs.store(
    group="dataset", name="cultural_captioning", node=DatasetConfig(type="cultural_captioning")
)

# Base Model
cs.store(group="model", name="base_model", node=ModelConfig)

# OpenAI models
cs.store(
    group="model",
    name="gpt-4",
    node=OpenaiModelConfig(name="gpt-4-vision-preview"),
)
cs.store(
    group="model",
    name="gpt-4-turbo",
    node=OpenaiModelConfig(name="gpt-4-turbo-2024-04-09"),
)

# Google models
cs.store(
    group="model",
    name="gemini-1.0",
    node=GoogleModelConfig(name="gemini-1.0-pro-vision"),
)
cs.store(
    group="model",
    name="gemini-1.5",
    node=GoogleModelConfig(name="gemini-1.5-pro-preview-0409"),
)

# Anthropic models
cs.store(
    group="model",
    name="claude-haiku",
    node=AnthropicModelConfig(name="claude-3-haiku-20240307"),
)
cs.store(
    group="model",
    name="claude-sonnet",
    node=AnthropicModelConfig(name="claude-3-sonnet-20240229"),
)
cs.store(
    group="model",
    name="claude-opus",
    node=AnthropicModelConfig(name="claude-3-opus-20240229"),
)

# HuggingFace models
cs.store(
    group="model",
    name="blip-2",
    node=HfModelConfig(
        name=f"Salesforce/blip2-opt-{II('model.size')}",
        size="6.7b",
        dtype="float16",
        model_cls=HfModel(_target_="transformers.Blip2ForConditionalGeneration.from_pretrained"),
        processor_cls=HfProcessor(_target_="transformers.Blip2Processor.from_pretrained"),
    ),
)

cs.store(
    group="model",
    name="instructblip",
    node=HfModelConfig(
        name=f"Salesforce/instructblip-vicuna-{II('model.size')}",
        size="7b",
        dtype="float16",
        model_cls=HfModel(
            _target_="transformers.InstructBlipForConditionalGeneration.from_pretrained"
        ),
        processor_cls=HfProcessor(_target_="transformers.InstructBlipProcessor.from_pretrained"),
    ),
)

cs.store(
    group="model",
    name="llava",
    node=HfModelConfig(
        name=f"llava-hf/llava-v1.6-{II('model.size')}-hf",
        size="mistral-7b",
        dtype="float16",
        model_cls=HfModel(
            _target_="transformers.LlavaNextForConditionalGeneration.from_pretrained",
            attn_implementation="flash_attention_2" if is_flashattn_2_supported() else "sdpa",
        ),
        processor_cls=HfProcessor(_target_="transformers.LlavaNextProcessor.from_pretrained"),
        strip_prompt=True,
    ),
)

cs.store(
    group="model",
    name="idefics2",
    node=HfModelConfig(
        name=f"HuggingFaceM4/idefics2-{II('model.size')}",
        size="8b",
        dtype="float16",
        model_cls=HfModel(
            _target_="transformers.AutoModelForVision2Seq.from_pretrained",
            attn_implementation="flash_attention_2" if is_flashattn_2_supported() else "sdpa",
        ),
        processor_cls=HfProcessor(_target_="transformers.AutoProcessor.from_pretrained"),
        strip_prompt=True,
    ),
)


# Callbacks
cs.store(group="callbacks", name="logging", node=LoggingCallbackConfig)
cs.store(group="callbacks", name="csv", node=SaveToCsvCallbackConfig)
cs.store(group="callbacks", name="wandb", node=WandbCallbackConfig)
