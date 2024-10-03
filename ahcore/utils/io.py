"""Input/ Output utils.

A part of the functions in this module are derived/taken from pytorch lightning template at:
https://github.com/ashleve/lightning-hydra-template
This template is licensed under the MIT License.

"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import warnings
from enum import Enum
from pathlib import Path
from types import FunctionType
from typing import Any, Optional, Sequence, Type

import hydra
import pytorch_lightning as pl
import rich
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from ahcore.models.base_jit_model import BaseAhcoreJitModel


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


def load_weights(model: LightningModule, config: DictConfig) -> LightningModule:
    """Load a model from a checkpoint file.

    Parameters
    ----------
    model: LightningModule
        The model to load the weights into.
    config : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    LightningModule
        The model loaded from the checkpoint file.
    """
    _model = getattr(model, "_model")
    if isinstance(_model, BaseAhcoreJitModel):
        return model
    if config.ckpt_path == "" or config.ckpt_path is None:
        raise ValueError("Checkpoint path not provided in config.")
    else:
        # Load checkpoint weights
        lit_ckpt = torch.load(config.ckpt_path)
        model.load_state_dict(lit_ckpt["state_dict"], strict=True)
    return model


def validate_checkpoint_paths(config: DictConfig) -> DictConfig:
    """
    Validate the checkpoint paths provided in the configuration files.

    Parameters
    ----------
    config: DictConfig
        Run configuration

    Returns
    -------
    DictConfig
    """
    # Extract paths with clear fallbacks
    checkpoint_path = config.get("ckpt_path")
    # this is not right and a bit hacky with the new models
    jit_path = config.get("lit_module", {}).get("model", {}).get("jit_path")
    # Validate configuration
    paths_defined = [path for path in [checkpoint_path, jit_path] if path]
    if len(paths_defined) == 0:
        logging.warning("No checkpoint or jit path provided in config.")
        return config
    elif len(paths_defined) > 1:
        raise RuntimeError("Checkpoint path and jit path cannot be defined simultaneously.")
    else:
        # Convert relative ckpt path to absolute path if necessary
        if checkpoint_path and not os.path.isabs(checkpoint_path):
            config.trainer.resume_from_checkpoint = pathlib.Path(hydra.utils.get_original_cwd()) / checkpoint_path
        return config


def get_git_hash() -> Optional[str]:
    try:
        # Check if we're in a git repository
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, capture_output=True, text=True)

        # Get the git hash
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # This will be raised if we're not in a git repo
        return None
    except FileNotFoundError:
        # This will be raised if git is not installed or not in PATH
        return None
