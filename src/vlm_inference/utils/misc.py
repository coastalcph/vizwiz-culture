import base64
import datetime
import logging

import randomname  # type: ignore
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def setup_logging() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s]  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )


def setup_config(config: DictConfig, print: bool = True) -> None:
    OmegaConf.resolve(config)
    del config._callback_dict

    if print:
        logger.info(f"\n{50*'-'}\n{OmegaConf.to_yaml(config)}{50*'-'}")


def as_dict(config: DictConfig):
    return OmegaConf.to_container(config, resolve=True)


def get_random_name():
    return f"{randomname.get_name()}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def read_image_as_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return image_bytes


def read_image_as_b64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def torch_dtype_from_str(dtype: str) -> torch.dtype:
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Invalid dtype: {dtype}")
    return DTYPE_MAP[dtype]


def is_flashattn_2_supported(device_id: int = 0):
    try:
        import flash_attn_2_cuda  # type: ignore # noqa: F401

        major, minor = torch.cuda.get_device_capability(device_id)

        # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
        is_sm8x = major == 8 and minor >= 0
        is_sm90 = major == 9 and minor == 0

        return is_sm8x or is_sm90
    except ImportError:
        logger.info("flash_attn_2_cuda not found. Disabling Flash Attention 2 support.")
        return False
