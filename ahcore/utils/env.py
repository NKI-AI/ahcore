from __future__ import annotations

import dotenv


def load_env(usecwd: bool = False, override: bool = True) -> None:
    """
    Load environment variables from `.env` file if it exists.
    Recursively searches for `.env` in all folders starting from working directory.

    Parameters
    ----------
    usecwd : bool, optional
        Wether to start from the current working directory to start searching for `.env` file, by default False.
    override : bool, optional
        Whether to override existing environment variables, by default True

    Returns
    -------
    None

    """
    dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv(usecwd=usecwd), override=override)
