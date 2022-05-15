import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src import utils

log = utils.get_logger(__name__)


def download(config: DictConfig) -> None:
    """Contains the data download pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Nothing
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Instantiate generator
    log.info(f"Instantiating downloader <{config.downloader._target_}>")
    downloader = hydra.utils.instantiate(config.downloader)
    

    # Generate the data
    log.info("Starting downloading!")
    downloader.download()
