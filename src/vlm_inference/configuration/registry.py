from hydra.core.config_store import ConfigStore
from omegaconf import II

from ..utils.misc import is_flashattn_2_supported
from .callbacks import (CostLoggingCallbackConfig, LoggingCallbackConfig,
                        SaveToCsvCallbackConfig, WandbCallbackConfig)
from .dataset import DatasetConfig
from .models import (AnthropicModelConfig, GoogleModelConfig, HfModel,
                     HfModelConfig, HfProcessor, ModelConfig,
                     OpenaiModelConfig, Pricing, RekaModelConfig)
from .run import RunConfig

cs = ConfigStore.instance()

# Base run
cs.store(name="base_config", node=RunConfig)

# Datasets
cs.store(
    group="dataset",
    name="captioning",
    node=DatasetConfig(_target_="vlm_inference.ImageCaptioningDataset"),
)
cs.store(
    group="dataset",
    name="cultural_captioning",
    node=DatasetConfig(_target_="vlm_inference.CulturalImageCaptioningDataset"),
)

# Base Model
cs.store(group="model", name="base_model", node=ModelConfig)

# OpenAI models
cs.store(
    group="model",
    name="gpt-4",
    node=OpenaiModelConfig(
        name="gpt-4-vision-preview",
        pricing=Pricing(usd_per_input_unit="10.00", usd_per_output_unit="30.00"),
    ),
)
cs.store(
    group="model",
    name="gpt-4-turbo",
    node=OpenaiModelConfig(
        name="gpt-4-turbo-2024-04-09",
        pricing=Pricing(usd_per_input_unit="10.00", usd_per_output_unit="30.00"),
    ),
)
cs.store(
    group="model",
    name="gpt-4o",
    node=OpenaiModelConfig(
        name="gpt-4o-2024-05-13",
        pricing=Pricing(usd_per_input_unit="5.00", usd_per_output_unit="15.00"),
    ),
)
cs.store(
    group="model",
    name="gpt-4o-mini",
    node=OpenaiModelConfig(
        name="gpt-4o-mini-2024-07-18",
        pricing=Pricing(usd_per_input_unit="0.15", usd_per_output_unit="0.60"),
    ),
)

# Google models
cs.store(
    group="model",
    name="gemini-1.0",
    node=GoogleModelConfig(
        name="gemini-1.0-pro-vision-001",
        pricing=Pricing(usd_per_input_unit="0.50", usd_per_output_unit="1.50"),
    ),
)
cs.store(
    group="model",
    name="gemini-1.5-pro",
    node=GoogleModelConfig(
        name="gemini-1.5-pro-preview-0514",
        pricing=Pricing(usd_per_input_unit="3.50", usd_per_output_unit="10.50"),
    ),
)
cs.store(
    group="model",
    name="gemini-1.5-flash",
    node=GoogleModelConfig(
        name="gemini-1.5-flash-preview-0514",
        pricing=Pricing(usd_per_input_unit="0.35", usd_per_output_unit="0.53"),
    ),
)

# Anthropic models
cs.store(
    group="model",
    name="claude-haiku",
    node=AnthropicModelConfig(
        name="claude-3-haiku-20240307",
        pricing=Pricing(usd_per_input_unit="0.25", usd_per_output_unit="1.25"),
    ),
)
cs.store(
    group="model",
    name="claude-sonnet",
    node=AnthropicModelConfig(
        name="claude-3-sonnet-20240229",
        pricing=Pricing(usd_per_input_unit="3.00", usd_per_output_unit="15.00"),
    ),
)
cs.store(
    group="model",
    name="claude-opus",
    node=AnthropicModelConfig(
        name="claude-3-opus-20240229",
        pricing=Pricing(usd_per_input_unit="15.00", usd_per_output_unit="75.00"),
    ),
)
cs.store(
    group="model",
    name="claude-3.5-sonnet",
    node=AnthropicModelConfig(
        name="claude-3-5-sonnet-20240620",
        pricing=Pricing(usd_per_input_unit="3.00", usd_per_output_unit="15.00"),
    ),
)

# Reka models
cs.store(
    group="model",
    name="reka-edge",
    node=RekaModelConfig(
        name="reka-edge-20240208",
        pricing=Pricing(usd_per_input_unit="0.4", usd_per_output_unit="1.0"),
    ),
)
cs.store(
    group="model",
    name="reka-flash",
    node=RekaModelConfig(
        name="reka-flash-20240226",
        pricing=Pricing(usd_per_input_unit="0.8", usd_per_output_unit="2.0"),
    ),
)
cs.store(
    group="model",
    name="reka-core",
    node=RekaModelConfig(
        name="reka-core-20240415",
        pricing=Pricing(usd_per_input_unit="10.0", usd_per_output_unit="25.0"),
    ),
)

# HuggingFace models
cs.store(
    group="model",
    name="blip2",
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

cs.store(
    group="model",
    name="paligemma",
    node=HfModelConfig(
        name=f"google/paligemma-{II('model.size')}",
        size="3b-mix-448",
        dtype="bfloat16",
        model_cls=HfModel(
            _target_="transformers.PaliGemmaForConditionalGeneration.from_pretrained",
            revision=II("model.dtype"),
        ),
        processor_cls=HfProcessor(_target_="transformers.AutoProcessor.from_pretrained"),
        strip_prompt=True,
    ),
)

cs.store(
    group="model",
    name="phi3-vision",
    node=HfModelConfig(
        name="microsoft/Phi-3-vision-128k-instruct",
        size="",  # not used
        dtype="bfloat16",
        model_cls=HfModel(
            _target_="transformers.AutoModelForCausalLM.from_pretrained",
            attn_implementation="flash_attention_2" if is_flashattn_2_supported() else "eager",
        ),
        processor_cls=HfProcessor(
            _target_="transformers.AutoProcessor.from_pretrained", use_fast=True
        ),
        strip_prompt=True,
    ),
)

cs.store(
    group="model",
    name="minicpm-llama3-v2.5",
    node=HfModelConfig(
        _target_="vlm_inference.CpmModel",
        name="openbmb/MiniCPM-Llama3-V-2_5",
        size="",  # not used
        dtype="float16",
        model_cls=HfModel(
            _target_="transformers.AutoModel.from_pretrained",
        ),
        processor_cls=HfProcessor(
            _target_="transformers.AutoTokenizer.from_pretrained", use_fast=True
        ),
    ),
)

cs.store(
    group="model",
    name="glm-4v",
    node=HfModelConfig(
        name=f"THUDM/glm-4v-{II('model.size')}",
        size="9b",
        dtype="bfloat16",
        model_cls=HfModel(
            _target_="transformers.AutoModelForCausalLM.from_pretrained",
        ),
        processor_cls=HfProcessor(
            _target_="vlm_inference.ChatGLMProcessor.from_pretrained", use_fast=True
        ),
        strip_prompt=True,
    ),
)


cs.store(group="callbacks", name="logging", node=LoggingCallbackConfig)
cs.store(group="callbacks", name="csv", node=SaveToCsvCallbackConfig)
cs.store(group="callbacks", name="wandb", node=WandbCallbackConfig)
cs.store(group="callbacks", name="cost_logging", node=CostLoggingCallbackConfig)
