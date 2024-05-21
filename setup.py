#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="vlm_inference",
    version="0.0.1",
    author="Phillip Rust",
    author_email="plip.rust@gmail.com",
    url="",
    description="",
    license="Apache 2.0",
    python_requires=">=3.10",
    install_requires=[
        "accelerate",
        "anthropic",
        "hydra-core",
        "hydra-submitit-launcher",
        "openai",
        "outlines<0.0.39",
        "pandas",
        "pillow",
        "pydantic",
        "randomname",
        "sentencepiece",
        "torch",
        "torchvision",
        "transformers",
        "vertexai",
        "wandb",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=True,
)
