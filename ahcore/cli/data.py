"""Module to write copy manifests files over to SCRATCH directory"""

from __future__ import annotations
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from rich.progress import Progress

from ahcore.cli import dir_path
from ahcore.utils.manifest import DataManager


def _quick_hash(file_path: Path, max_bytes: int = 10**6) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        block = f.read(max_bytes)
        hasher.update(block)
    return hasher.hexdigest()


def copy_data(args: argparse.Namespace) -> None:
    manifest_url = args.manifest_uri
    base_dir = args.base_dir
    dataset_name = args.dataset_name
    target_dir = os.environ.get("SCRATCH", None)

    # Initialize a counter for the total data size
    total_size = 0

    if target_dir is None or not os.access(target_dir, os.W_OK):
        print("Please set the SCRATCH environment variable to a writable directory.")
        sys.exit(1)

    with DataManager(manifest_url) as dm:
        all_records = dm.get_records_by_split(args.manifest_name, args.split_name, split_category=None)
        with Progress() as progress:
            task = progress.add_task("[cyan]Copying...")
            for patient in all_records:
                for image in patient.images:
                    image_fn = image.filename

                    get_from = base_dir / image_fn
                    write_to = Path(target_dir) / dataset_name / "images" / image_fn

                    accompanying_folder_write_to, accompanying_folder_get_from = None, None
                    if get_from.suffix == ".mrxs":
                        accompanying_folder_get_from = get_from.parent / get_from.stem
                        if not accompanying_folder_get_from.is_dir():
                            raise ValueError(
                                f"Image {image_fn} does not have an accompanying folder, which is expected for mrxs images"
                            )
                        accompanying_folder_write_to = write_to.parent / write_to.stem

                    write_to.parent.mkdir(parents=True, exist_ok=True)
                    if write_to.exists():
                        # compute the hash of previous and new file
                        old_hash = _quick_hash(write_to)
                        new_hash = _quick_hash(get_from)
                        if old_hash == new_hash:
                            # Skip if they are the same
                            progress.console.log("Skipping (image) file as it already exists: {}".format(image_fn))
                            progress.update(task, advance=1)
                            continue

                    total_size += get_from.stat().st_size

                    # Copy file from get_from to write_to
                    shutil.copy(get_from, write_to)
                    if accompanying_folder_get_from is not None:
                        shutil.copytree(accompanying_folder_get_from, accompanying_folder_write_to, dirs_exist_ok=True)
                        total_size += accompanying_folder_get_from.stat().st_size
                    progress.update(task, advance=1)

                    # copy mask and annotations
                    if args.mask_dir is not None or args.annotations_dir is not None:
                        for mask in image.masks:
                            mask_fn = mask.filename
                            get_from = args.mask_dir / mask_fn
                            write_to = Path(target_dir) / dataset_name / "masks" / mask_fn
                            write_to.parent.mkdir(parents=True, exist_ok=True)
                            if write_to.exists():
                                # compute the hash of previous and new file
                                old_hash = _quick_hash(write_to)
                                new_hash = _quick_hash(get_from)
                                if old_hash == new_hash:
                                    # Skip if they are the same
                                    progress.console.log(
                                        "Skipping (mask) file as it already exists: {}".format(mask_fn)
                                    )
                                    progress.update(task, advance=1)
                                    continue

                            total_size += get_from.stat().st_size
                            shutil.copy(get_from, write_to)
                            progress.update(task, advance=1)

                        for annotation in image.annotations:
                            annotation_fn = annotation.filename
                            get_from = args.annotations_dir / annotation_fn
                            write_to = Path(target_dir) / dataset_name / "annotations" / annotation_fn
                            write_to.parent.mkdir(parents=True, exist_ok=True)
                            if write_to.exists():
                                # compute the hash of previous and new file
                                old_hash = _quick_hash(write_to)
                                new_hash = _quick_hash(get_from)
                                if old_hash == new_hash:
                                    # Skip if they are the same
                                    progress.console.log(
                                        "Skipping (annotation) file as it already exists: {}".format(annotation_fn)
                                    )
                                    progress.update(task, advance=1)
                                    continue

                            total_size += get_from.stat().st_size
                            shutil.copy(get_from, write_to)
                            progress.update(task, advance=1)

                    # copy features
                    if args.features_dir is not None:
                        if args.feature_version is not None:
                            feature = dm.get_image_features_by_image_and_feature_version(
                                image.id, args.feature_version
                            )  # there should only be one
                            feature_fn = feature.filename
                            get_from = args.features_dir / feature_fn
                            write_to = Path(target_dir) / dataset_name / "features" / feature_fn
                            write_to.parent.mkdir(parents=True, exist_ok=True)
                            if write_to.exists():
                                # compute the hash of previous and new file
                                old_hash = _quick_hash(write_to)
                                new_hash = _quick_hash(get_from)
                                if old_hash == new_hash:
                                    # Skip if they are the same
                                    progress.console.log(
                                        "Skipping (feature) file as it already exists: {}".format(feature_fn)
                                    )
                                    progress.update(task, advance=1)
                                    continue

                            total_size += get_from.stat().st_size
                            shutil.copy(get_from, write_to)
                            progress.update(task, advance=1)
                        else:
                            raise ValueError(
                                "Feature version is not provided, but features directory is provided. Please provide both to copy features."
                            )

    progress.console.log("Total data size copied: {:.2f} GB".format(total_size / 1024**3))


def register_parser(
    parser: argparse._SubParsersAction[Any],  # pylint: disable=unsubscriptable-object
) -> None:  # pylint: disable=E1136
    """Register inspect commands to a root parser."""
    data_parser = parser.add_parser("data", help="Data utilities")
    data_subparsers = data_parser.add_subparsers(help="Data subparser")
    data_subparsers.required = True
    data_subparsers.dest = "subcommand"

    _parser: argparse.ArgumentParser = data_subparsers.add_parser(
        "copy-data-from-manifest",
        help="Copy the data to a different drive based on the manifest. "
        "The data will be copied over to $SCRATCH / DATASET_NAME",
    )

    _parser.add_argument(
        "manifest_uri",
        type=str,
        help="URI that refers to the sqlalchemy supported database path. If using an sqlite database this looks like 'sqlite:///your_database_path'.",
    )
    _parser.add_argument(
        "manifest_name",
        type=str,
        help="Name of the manifest to copy the data from.",
    )
    _parser.add_argument(
        "split_name",
        type=str,
        help="Name of the split in the database to copy the data from.",
    )
    _parser.add_argument(
        "base_dir",
        type=dir_path(require_writable=False),
        help="Directory to which the paths defined in the manifest are relative to.",
    )
    _parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to copy the data to. The data will be copied over to $SCRATCH / DATASET_NAME",
    )

    _parser.add_argument(
        "--mask_dir",
        type=dir_path(require_writable=False),
        help="Directory to which the masks paths defined in the manifest are relative to.",
        default=None,
    )

    _parser.add_argument(
        "--annotations_dir",
        type=dir_path(require_writable=False),
        help="Directory to which the annotations paths defined in the manifest are relative to.",
        default=None,
    )

    _parser.add_argument(
        "--features_dir",
        type=dir_path(require_writable=False),
        help="Directory to which the features paths defined in the manifest are relative to.",
        default=None,
    )

    _parser.add_argument(
        "--feature_version",
        type=dir_path(require_writable=False),
        help="Version of the features to copy.",
        default=None,
    )

    _parser.set_defaults(subcommand=copy_data)
