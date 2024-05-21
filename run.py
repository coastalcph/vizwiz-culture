import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from vlm_inference import run_engine, setup_config, setup_logging


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    setup_logging()
    setup_config(config)

    run_engine(
        model=instantiate(config.model),
        dataset=instantiate(config.dataset),
        callbacks=[instantiate(c) for c in config.callbacks],
    )


if __name__ == "__main__":
    main()
