from __future__ import annotations

from pathlib import Path
from typing import Optional

import dotenv


def load_env(working_dir: Optional[Path | str] = None, override: bool = True) -> None:
    """
    Load environment variables from `.env` file if it exists.
    Recursively searches for `.env` in all folders starting from working directory.

    Parameters
    ----------
    working_dir : Optional[Path]
        Working directory to start searching for `.env` file.
    override : bool, optional
        Whether to override existing environment variables, by default True

    Returns
    -------
    None

    """
    dotenv.load_dotenv(dotenv_path=working_dir, override=override)
