from dataclasses import dataclass
from typing import List

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig

from vlm_inference.configuration import BaseConfig, CallbackConfig, DatasetConfig, ModelConfig
from vlm_inference.engine import Engine
from vlm_inference.utils import setup_config, setup_logging


@dataclass
class RunConfig(BaseConfig):
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    callbacks: List[CallbackConfig] = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=RunConfig)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    setup_logging()
    setup_config(config)

    Engine(
        model=instantiate(config.model),
        dataset=instantiate(config.dataset),
        callbacks=[instantiate(c) for c in config.callbacks]
    ).run()

if __name__ == "__main__":
    main()
