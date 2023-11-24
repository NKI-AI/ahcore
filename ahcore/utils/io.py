"""Input/ Output utils.

A part of the functions in this module are derived/taken from pytorch lightning template at:
https://github.com/ashleve/lightning-hydra-template
This template is licensed under the MIT License.

"""
from __future__ import annotations

import logging
import os
import warnings
from enum import Enum
from pathlib import Path
from types import FunctionType
from typing import Any, Optional, Sequence, Type

import pytorch_lightning as pl
import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from pytorch_lightning.utilities import rank_zero_only  # type: ignore[attr-defined]


def get_logger(name: str = __name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


logger = get_logger(__name__)


def debug_function(x: int) -> int:
    """
    Function to use for debugging (e.g. github workflow testing)
    :param x:
    :return: x^2
    """
    return x**2


def validate_config(cfg: Any) -> None:
    if isinstance(cfg, ListConfig):
        for x in cfg:
            validate_config(x)
    elif isinstance(cfg, DictConfig):
        for name, v in cfg.items():
            if name == "hydra":
                logger.warning("Skipped validating hydra native configs")
                continue
            try:
                validate_config(v)
            except InterpolationKeyError:
                logger.warning("Skipped validating %s: %s", name, str(v))
                continue


@rank_zero_only  # type: ignore[misc]
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "experiment",
        "transforms",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Arguments
    ---------
    config : DictConfig
        Configuration composed by Hydra.
    fields : Sequence[str], optional
        Determines which main fields from config will be printed and in what order.
    resolve : bool, optional
        Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    logger = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        logger.info("Disabling python warnings <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("enforce_tags") and (not config.get("tags") or config.get("tags") == ["dev"]):
        logger.info(
            "Running in experiment mode without tags specified"
            "Use `python run.py experiment=some_experiment tags=['some_tag',...]`, or change it in the experiment yaml"
        )
        logger.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        logger.info("Forcing debugger friendly configuration <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only  # type:ignore[misc]
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {
        "model": config["lit_module"],
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model/params/non_trainable": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "datamodule": config["datamodule"],
        "trainer": config["trainer"],
    }

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def fullname(obj: Any) -> str:
    if isinstance(obj, type) or isinstance(obj, FunctionType):
        cls = obj
    else:  # if obj is an instance, get its class
        cls = type(obj)

    module = cls.__module__
    if module is None or module == str.__class__.__module__:  # don't want to return 'builtins'
        return cls.__name__
    return module + "." + cls.__name__


def get_enum_key_from_value(value: str, enum_class: Type[Enum]) -> Optional[str]:
    for enum_member in enum_class:
        if enum_member.value == value:
            return enum_member.name
    return None


def get_cache_dir() -> Path:
    return Path(os.environ.get("SCRATCH", "/tmp")) / "ahcore_cache"
