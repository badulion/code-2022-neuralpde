import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src import utils
from src.utils.pde_generator import GenerationUtility

log = utils.get_logger(__name__)


def generate(config: DictConfig) -> None:
    """Contains the data generation pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Nothing
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Instantiate generator
    log.info(f"Instantiating generator <{config.equation._target_}>")
    generator: GenerationUtility = hydra.utils.instantiate(config.equation)
    

    # Generate the data
    log.info("Starting generating!")
    generator.generate()
