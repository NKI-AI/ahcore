"""Ahcore Command-line interface. This is the file which builds the main parser."""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Callable


def dir_path(require_writable: bool = False) -> Callable[[str], pathlib.Path]:
    def check_dir_path(path: str) -> pathlib.Path:
        """Check if the path is a valid and (optionally) writable directory.

        Parameters
        ----------
        path : str

        Returns
        -------
        pathlib.Path
            The path as a pathlib.Path object.
        """
        _path = pathlib.Path(path)
        if _path.is_dir():
            if require_writable:
                if os.access(_path, os.W_OK):
                    return _path
                else:
                    raise argparse.ArgumentTypeError(f"{path} is not a writable directory.")
            else:
                return _path
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory.")

    return check_dir_path


def file_path(path: str) -> pathlib.Path:
    """Check if the path is a valid file.

    Parameters
    ----------
    path : str

    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.

    """
    _path = pathlib.Path(path)
    if _path.is_file():
        return _path
    raise argparse.ArgumentTypeError(f"{path} is not a valid file.")


def main() -> None:
    """
    Main entrypoint for the CLI command of ahcore.
    """
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="Possible ahcore CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    # Prevent circular import
    from ahcore.cli.data import register_parser as register_data_subcommand

    # Data related commands.
    register_data_subcommand(root_subparsers)

    # Prevent circular import
    from ahcore.cli.tiling import register_parser as register_tiling_subcommand

    # Tiling related commands
    register_tiling_subcommand(root_subparsers)

    args = root_parser.parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main()
