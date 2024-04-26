import logging
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from dataset import get_dataset
from llm_wrapper import SaveStrategy, get_llm_wrapper


@dataclass
class RunConfig:
    model_name: str = field(
        default=MISSING,
        metadata={"help": "The name of the model to use. Must be a key in MODEL_CONFIGS."},
    )
    dataset_path: str = field(
        default=MISSING,
        metadata={"help": "The path to the dataset to use. Should be a folder containing images."},
    )
    dataset_type: str = field(
        default="captioning",
        metadata={"help": "The type of the dataset to use. Options are 'captioning'."},
    )
    template_name: str = field(
        default="default",
        metadata={"help": "Name of the template to use for the prompt."},
    )
    parse_json: bool = field(
        default=False,
        metadata={"help": "Whether to parse the model outputs as JSON."},
    )
    model_size: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The size of the model to use, e.g. '7b'. Overrides the default model size defined "
                " in MODEL_CONFIGS. Only used for HuggingFace models."
            )
        },
    )
    save_strategy: SaveStrategy = field(
        default=SaveStrategy.LOCAL,
        metadata={
            "help": "The strategy to use for saving model outputs and evaluation results, either 'LOCAL' or 'WANDB'."
        },
    )
    save_path: Optional[str] = field(
        default="./outputs",
        metadata={"help": "The path to save model outputs to. Only used with SaveStrategy.LOCAL."},
    )


cs = ConfigStore.instance()
cs.store(name="base_config", node=RunConfig)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    # Reformat logger
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s]  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    print(f"Run config:\n{'-'*20}\n{OmegaConf.to_yaml(config)}{'-'*20}\n")

    llm = get_llm_wrapper(config)

    dataset = get_dataset(config)

    llm.run_inference(dataset)


if __name__ == "__main__":
    main()
