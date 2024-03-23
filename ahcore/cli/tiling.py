"""Utility to create tiles from the TCGA FFPE H&E slides.

Many models uses 0.5um/pixel at 224 x 224 size.
"""

from __future__ import annotations

import argparse
import io
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat
from typing import Any, Generator, Literal, NamedTuple

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageCms
from dlup import SlideImage
from dlup.data.dataset import TiledWsiDataset
from dlup.experimental_backends import ImageBackend
from dlup.tiling import GridOrder, TilingMode
from PIL import Image
from pydantic import BaseModel
from rich.progress import Progress

from ahcore.cli import dir_path, file_path
from ahcore.writers import H5FileImageWriter

logger = getLogger(__name__)


def read_mask(path: Path) -> npt.NDArray[np.int_]:
    return iio.imread(path)[..., 0]


def _to_pil_compression(compression: Literal["jpg", "png", "none"]) -> Literal["JPEG", "PNG"] | None:
    _compression: Literal["JPEG", "PNG"] | None = None
    if compression == "jpg":
        _compression = "JPEG"
    elif compression == "png":
        _compression = "PNG"
    elif compression == "none":
        pass
    else:
        raise ValueError(f"Invalid compression {compression}")

    return _compression


class SlideImageMetaData(BaseModel):
    """Metadata of a whole slide image."""

    path: Path
    mpp: float
    aspect_ratio: float
    magnification: float | None
    size: tuple[int, int]
    vendor: str | None

    @classmethod
    def from_dataset(cls, dataset: TiledWsiDataset) -> "SlideImageMetaData":
        _relevant_keys = ["aspect_ratio", "magnification", "mpp", "size", "vendor"]
        return cls(
            **{
                "path": dataset.path,
                **{key: getattr(dataset.slide_image, key) for key in _relevant_keys},
            }
        )

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for k, v in self.model_dump().items():
            yield k, v


class TileMetaData(BaseModel):
    coordinates: tuple[int, int]
    region_index: int
    grid_local_coordinates: tuple[int, int]
    grid_index: int

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for k, v in self.model_dump().items():
            yield k, v


class DatasetConfigs(BaseModel):
    """Configurations of the TiledWsiDataset dataset"""

    mpp: float
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int]
    tile_mode: str
    crop: bool
    mask_threshold: float
    grid_order: str
    color_profile_applied: bool = False


class Thumbnail(NamedTuple):
    """Thumbnail of a slide image."""

    thumbnail: npt.NDArray[np.uint8]
    mask: npt.NDArray[np.uint8] | None
    overlay: npt.NDArray[np.uint8] | None


def _save_thumbnail(
    image_fn: Path,
    dataset_cfg: DatasetConfigs,
    mask: npt.NDArray[np.int_] | None,
    store_overlay: bool = False,
) -> Thumbnail:
    """
    Saves a thumbnail of the slide, including the filtered tiles and the mask itself.

    In case apply_color_profile is set, the color profile will be applied to the output, otherwise
    it will be in the 'icc_profile' attribute of the thumbnail and overlay.

    Computing the overlay is a slow process.

    Parameters
    ----------
    image_fn : Path
        Path to the whole slide image file.
    dataset_cfg : DatasetConfigs
        Dataset configurations.
    mask : np.ndarray | None
        Binary mask used to filter each tile.
    store_overlay : bool
        Store the overlay and mask as well.

    Returns
    -------
    Thumbnail
        All relevant information

    """
    # Let's figure out an appropriate target_mpp, our thumbnail has to be at least 512px in size
    _scaling = 1.0
    with SlideImage.from_file_path(image_fn, backend=ImageBackend.OPENSLIDE) as slide_image:
        scaled_size = max(slide_image.size)
        while scaled_size >= 1024:
            scaled_size = max(slide_image.get_scaled_size(_scaling))
            _scaling /= 2
        _scaling *= 2
        target_mpp = slide_image.get_mpp(_scaling)
        target_size = slide_image.get_scaled_size(_scaling)

        thumbnail_io = io.BytesIO()
        thumbnail = slide_image.get_thumbnail(target_size)
        # If the color_profile is not applied, we need to add it to the metadata of the thumbnail
        if slide_image.color_profile:
            to_profile = PIL.ImageCms.createProfile("sRGB")
            intent = PIL.ImageCms.getDefaultIntent(slide_image.color_profile)
            rgb_color_transform = PIL.ImageCms.buildTransform(
                slide_image.color_profile, to_profile, "RGB", "RGB", intent, 0
            )
            PIL.ImageCms.applyTransform(thumbnail, rgb_color_transform, True)

        thumbnail.convert("RGB").save(thumbnail_io, format="JPEG", quality=75)
        _thumbnail = np.frombuffer(thumbnail_io.getvalue(), dtype="uint8")

    if not store_overlay:
        return Thumbnail(thumbnail=_thumbnail, mask=mask.astype(np.uint8) if mask is not None else None, overlay=None)

    tile_size = (32, 32)
    dataset = TiledWsiDataset.from_standard_tiling(
        image_fn,
        target_mpp,
        tile_size,
        (0, 0),
        mask=mask,
        crop=False,
        tile_mode=TilingMode.overflow,
        mask_threshold=dataset_cfg.mask_threshold,
        apply_color_profile=dataset_cfg.color_profile_applied,
        backend=ImageBackend.OPENSLIDE,
    )

    scaled_region_view = dataset.slide_image.get_scaled_view(dataset.slide_image.get_scaling(target_mpp))
    region_size = tuple(scaled_region_view.size)
    overlay = Image.new("RGBA", (region_size[0], region_size[1]), (255, 255, 255, 255))

    overlay_io = io.BytesIO()
    for d in dataset:
        tile = d["image"]
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        overlay.paste(tile, (box[0], box[1]))
        # You could uncomment this to plot the boxes of the grid as well, but this can quickly become crowded.
        # draw = ImageDraw.Draw(background)
        # draw.rectangle(box, outline="red")

    # If the below were true, it would already be embedded into the tile.
    if not dataset_cfg.color_profile_applied:
        overlay.info["icc_profile"] = dataset.slide_image.color_profile.tobytes()  # type: ignore

    overlay.convert("RGB").save(overlay_io, format="JPEG", quality=75)

    _thumbnail = np.frombuffer(thumbnail_io.getvalue(), dtype="uint8")
    _overlay = np.frombuffer(overlay_io.getvalue(), dtype="uint8")

    return Thumbnail(thumbnail=_thumbnail, mask=mask.astype(np.uint8) if mask is not None else None, overlay=_overlay)


def create_slide_image_dataset(
    slide_image_path: Path,
    mask: SlideImage | npt.NDArray[np.int_] | None,
    cfg: DatasetConfigs,
    overwrite_mpp: tuple[float, float] | None = None,
) -> TiledWsiDataset:
    """
    Initializes and returns a slide image dataset.

    Parameters
    ----------
    slide_image_path : Path
        Path to a whole slide image file.
    mask : np.ndarray | None
        Binary mask used to filter each tile.
    cfg : DatasetConfigs
        Dataset configurations.
    overwrite_mpp : tuple[float, float] | None
        Tuple of (mpp_x, mpp_y) used to overwrite the mpp of the loaded slide image.

    Returns
    -------
    TiledROIsSlideImageDataset
        Initialized slide image dataset.

    """

    return TiledWsiDataset.from_standard_tiling(
        path=slide_image_path,
        mpp=cfg.mpp,
        tile_size=cfg.tile_size,
        tile_overlap=cfg.tile_overlap,
        grid_order=GridOrder[cfg.grid_order],
        tile_mode=TilingMode[cfg.tile_mode],
        crop=cfg.crop,
        mask=mask,
        mask_threshold=cfg.mask_threshold,
        overwrite_mpp=overwrite_mpp,
        apply_color_profile=cfg.color_profile_applied,
        backend=ImageBackend.OPENSLIDE,
    )


def _generator(
    dataset: TiledWsiDataset, quality: int | None, compression: Literal["JPEG", "PNG"] | None
) -> Generator[Any, Any, Any]:
    for idx in range(len(dataset)):
        # TODO: This should be working iterating over the dataset
        sample = dataset[idx]
        tile: PIL.Image.Image = sample["image"]
        # If we just cast the PIL.Image to RGB, the alpha channel is set to black
        # which is a bit unnatural if you look in the image pyramid where it would be white in lower resolutions
        # this is why we take the following approach.
        output = PIL.Image.new("RGB", tile.size, (255, 255, 255))  # Create a white background
        output.paste(tile, mask=tile.split()[3])  # Paste the image using the alpha channel as mask
        if compression in ["JPEG", "PNG"]:  # This is JPG
            buffered = io.BytesIO()
            _quality = None if compression == "PNG" else quality
            output.convert("RGB").save(buffered, format=compression, quality=quality)
            array = np.frombuffer(buffered.getvalue(), dtype="uint8")
        else:  # No compression
            array = np.asarray(tile)

        # Now we have the image bytes
        coordinates = sample["coordinates"]

        yield [coordinates], array[np.newaxis, :]


def save_tiles(
    dataset: TiledWsiDataset,
    h5_writer: H5FileImageWriter,
    compression: Literal["jpg", "png", "none"],
    quality: int | None = 85,
) -> None:
    """
    Saves the tiles in the given image slide dataset to disk.

    Parameters
    ----------
    dataset : TiledROIsSlideImageDataset
        The image slide dataset containing tiles of a single whole slide image.
    h5_writer : H5FileImageWriter
        The H5 writer to write the tiles to.
    compression : Literal["jpg", "png", "none"]
        Either "jpg", "png" or "none".
    quality : int | None
        If not None, the compression quality of the saved tiles in jpg, otherwise png

    """
    _compression = _to_pil_compression(compression)
    _quality = None if _compression != "JPEG" else quality

    generator = _generator(dataset, _quality, _compression)
    h5_writer.consume(generator)


def _tiling_pipeline(
    image_path: Path,
    mask_path: Path | None,
    output_file: Path,
    dataset_cfg: DatasetConfigs,
    compression: Literal["png", "jpg", "none"],
    quality: int,
    save_thumbnail: bool = False,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # TODO: Come up with a way to inject the mask later on as well.
        mask = read_mask(mask_path) if mask_path else None
        dataset = create_slide_image_dataset(
            slide_image_path=image_path,
            mask=mask,
            cfg=dataset_cfg,
        )
        _scaling = dataset.slide_image.get_scaling(dataset_cfg.mpp)

        color_profile = None
        if not dataset_cfg.color_profile_applied and dataset.slide_image.color_profile:
            color_profile = dataset.slide_image.color_profile.tobytes()

        extra_metadata = {
            "color_profile_applied": dataset_cfg.color_profile_applied,
        }

        h5_writer = H5FileImageWriter(
            filename=output_file,
            size=dataset.slide_image.get_scaled_slide_bounds(_scaling)[1],
            mpp=dataset_cfg.mpp,
            tile_size=dataset_cfg.tile_size,
            tile_overlap=dataset_cfg.tile_overlap,
            num_samples=len(dataset),
            is_compressed_image=compression != "none",
            color_profile=color_profile,
            extra_metadata=extra_metadata,
        )
        save_tiles(dataset, h5_writer, compression=compression, quality=quality)
        if save_thumbnail:
            thumbnail_obj = _save_thumbnail(image_path, dataset_cfg, mask)

            description = "thumbnail"
            images = [
                ("thumbnail", thumbnail_obj.thumbnail),
            ]
            if thumbnail_obj.mask is not None:
                description += ", mask"
                images.append(("mask", thumbnail_obj.mask))

            if thumbnail_obj.overlay:
                description += ", overlay"
                images.append(("overlay", thumbnail_obj.overlay))

            h5_writer.add_associated_images(
                images=tuple(images),
                description=description,
            )

    except Exception as e:
        logger.error(f"Failed: {image_path} with exception {e}")
        return

    logger.debug("Working on %s. Writing to %s", image_path, output_file)


def _wrapper(
    dataset_cfg: DatasetConfigs,
    compression: Literal["jpg", "png", "none"],
    quality: int,
    save_thumbnail: bool,
    args: tuple[Path, Path, Path],
) -> None:
    image_path, mask_path, output_file = args
    return _tiling_pipeline(
        image_path=image_path,
        mask_path=mask_path,
        output_file=output_file,
        dataset_cfg=dataset_cfg,
        compression=compression,
        quality=quality,
        save_thumbnail=save_thumbnail,
    )


def _do_tiling(args: argparse.Namespace) -> None:
    images_list: list[tuple[Path, Path | None, Path]] = []

    with open(args.file_list, "r") as file_list:
        for line in file_list:
            if line.startswith("#"):
                continue
            image_file, mask_file, output_filename = line.split(",")
            if (args.output_directory / "data" / Path(output_filename.strip())).is_file() and args.simple_check:
                continue

            mask_fn = Path(mask_file.strip()) if mask_file != "" else None
            images_list.append(
                (
                    Path(image_file.strip()),
                    mask_fn,
                    args.output_directory / "data" / Path(output_filename.strip()),
                )
            )

    logger.info(f"Number of slides: {len(images_list)}")
    logger.info(f"Output directory: {args.output_directory}")
    logger.info("Tiling...")

    save_dir_data = args.output_directory / "data"
    save_dir_data.mkdir(parents=True, exist_ok=True)

    crop = False
    tile_mode = TilingMode.overflow
    tile_size = (args.tile_size, args.tile_size)
    tile_overlap = (args.tile_overlap, args.tile_overlap)
    mpp = args.mpp
    mask_threshold = args.mask_threshold

    dataset_cfg = DatasetConfigs(
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_mode=tile_mode,
        crop=crop,
        mask_threshold=mask_threshold,
        grid_order="C",
        color_profile_applied=args.apply_color_profile,
    )

    logger.info(f"Dataset configurations: {pformat(dataset_cfg)}")
    if args.num_workers > 0:
        # Create a partially applied function with dataset_cfg
        partial_wrapper = partial(_wrapper, dataset_cfg, args.compression, args.quality, args.save_thumbnail)

        with Progress() as progress:
            task = progress.add_task("[cyan]Tiling...", total=len(images_list))
            with Pool(processes=args.num_workers) as pool:
                for _ in pool.imap_unordered(partial_wrapper, images_list):
                    progress.update(task, advance=1)
    else:
        with Progress() as progress:
            task = progress.add_task("[cyan]Tiling...", total=len(images_list))
            for idx, (image_path, mask_path, output_file) in enumerate(images_list):
                _tiling_pipeline(
                    image_path=image_path,
                    mask_path=mask_path,
                    output_file=output_file,
                    dataset_cfg=dataset_cfg,
                    compression=args.compression,
                    quality=args.quality,
                    save_thumbnail=args.save_thumbnail,
                )

                progress.update(task, advance=1)


def register_parser(parser: argparse._SubParsersAction[Any]) -> None:
    """Register inspect commands to a root parser."""
    tiling_parser = parser.add_parser("tiling", help="Tiling utilities")
    tiling_subparsers = tiling_parser.add_subparsers(help="Tiling subparser")
    tiling_subparsers.required = True
    tiling_subparsers.dest = "subcommand"

    _parser: argparse.ArgumentParser = tiling_subparsers.add_parser(
        "tile-to-h5",
        help="Tiling WSI images to h5",
    )

    # Assume a comma separated format from image_file,mask_file
    _parser.add_argument(
        "--file-list",
        type=file_path,
        required=True,
        help="Path to the file list. Each comma-separated line is of the form `<image_fn>,<mask_fn>,<output_directory>`"
        " where the output directory is with respect to --output-directory. mask_fn can be empty.",
    )
    _parser.add_argument(
        "--output-directory",
        type=dir_path(require_writable=True),
        required=True,
        help="Path to the output directory.",
    )
    _parser.add_argument(
        "--mpp",
        type=float,
        required=True,
        help="Resolution (microns per pixel) at which the slides should be tiled.",
    )
    _parser.add_argument("--tile-size", type=int, required=True, help="Size of the tiles in pixels.")
    _parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Overlap of the tile in pixels (default=0).",
    )
    _parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.6,
        help="0 every tile is discarded, 1 requires the whole tile to be foreground (default=0.6).",
    )
    _parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers to use for tiling. 0 disables the tiling (default: 8)",
    )
    _parser.add_argument(
        "--save-thumbnail",
        action="store_true",
        help="Save a thumbnail of the slide, including the filtered tiles and the mask itself.",
    )
    _parser.add_argument(
        "--simple-check",
        action="store_true",
        help="Filter the list based on if the h5 images already exist.",
    )
    _parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="Quality of the saved tiles in jpg (default: 80)",
    )

    _parser.add_argument(
        "--compression",
        type=str,
        choices=["jpg", "png", "none"],
        default="jpg",
        help="The compression to use, either jpg, png or none (default=jpg)",
    )

    _parser.add_argument(
        "--apply-color-profile",
        action="store_true",
        help="Apply the color profile of the slide in the tiles. "
        "This will also remove the color profile from the attributes.",
    )

    _parser.set_defaults(subcommand=_do_tiling)
