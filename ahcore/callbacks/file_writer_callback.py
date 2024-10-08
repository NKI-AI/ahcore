from __future__ import annotations

from pathlib import Path
from typing import Type

from dlup.data.dataset import TiledWsiDataset
from dlup.tiling import Grid

from ahcore.callbacks.abstract_writer_callback import AbstractWriterCallback
from ahcore.callbacks.converters.common import ConvertCallbacks
from ahcore.lit_module import AhCoreLightningModule
from ahcore.utils.callbacks import get_output_filename as get_output_filename_
from ahcore.utils.data import DataDescription, GridDescription
from ahcore.utils.io import get_logger
from ahcore.utils.types import DataFormat, InferencePrecision, NormalizationType
from ahcore.writers import Writer

logger = get_logger(__name__)


class WriteFileCallback(AbstractWriterCallback):
    def __init__(
        self,
        writer_class: Type[Writer],
        queue_size: int,
        max_concurrent_queues: int,
        dump_dir: Path,
        normalization_type: str = NormalizationType.LOGITS,
        precision: str = InferencePrecision.FP32,
        callbacks: list[ConvertCallbacks] | None = None,
        data_format: str = DataFormat.IMAGE,
    ) -> None:
        """
        Callback to write predictions to H5 files. This callback is used to write whole-slide predictions to single H5
        files in a separate thread.

        Parameters
        ----------
        writer_class : Writer
            The writer class to use to write the predictions to e.g. H5 files.
        queue_size : int
            The maximum number of items to store in the queue (i.e. tiles).
        max_concurrent_queues : int
            The maximum number of concurrent writers.
        dump_dir : pathlib.Path
            The directory to dump the H5 files to.
        normalization_type : str
            The normalization type to use for the predictions. One of "sigmoid", "softmax" or "logits".
        precision : str
            The precision to use for the predictions. One of "float16", "float32" or "uint8".
        """
        self._current_filename = None
        self._dump_dir = Path(dump_dir)

        self._writer_class: Type[Writer] = writer_class
        self._suffix = ".cache"
        self._normalization_type: NormalizationType = NormalizationType(normalization_type)
        self._precision: InferencePrecision = InferencePrecision(precision)
        self._data_format = DataFormat(data_format)

        super().__init__(
            writer_class=writer_class,
            dump_dir=self._dump_dir,
            queue_size=queue_size,
            max_concurrent_queues=max_concurrent_queues,
            data_key="prediction",
            normalization_type=normalization_type,
            precision=precision,
            callbacks=callbacks,
        )

        self._dataset_index = 0

    @property
    def dump_dir(self) -> Path:
        return self._dump_dir

    def get_output_filename(self, pl_module: AhCoreLightningModule, filename: str) -> Path:
        output_filename = get_output_filename_(
            self.dump_dir,
            Path(filename),
            model_name=str(pl_module.name),
            counter=f"{pl_module.current_epoch}_{pl_module.validation_counter}",
        )
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        return output_filename

    def build_writer_class(self, pl_module: AhCoreLightningModule, stage: str, filename: str) -> Writer:
        output_filename = self.get_output_filename(pl_module, filename)
        # TODO: need to make a path builder as this is shared with the output_filename method.
        link_fn = (
            self.dump_dir
            / "outputs"
            / f"{pl_module.name}"
            / f"{pl_module.current_epoch}_{pl_module.validation_counter}"
            / "image_cache_link.txt"
        )
        link_fn.parent.mkdir(parents=True, exist_ok=True)

        with open(link_fn, "a" if link_fn.is_file() else "w") as file:
            file.write(f"{filename},{output_filename}\n")

        size, mpp, tile_size, tile_overlap, num_samples, grid = self._get_writer_data_args(
            pl_module, data_format=self._data_format, stage=stage
        )

        writer = self._writer_class(
            output_filename,
            size=size,  # --> (num_samples,1)
            mpp=mpp,
            tile_size=tile_size,  # --> (1,1)
            tile_overlap=tile_overlap,
            num_samples=num_samples,
            color_profile=None,
            data_format=self._data_format,
            progress=None,
            precision=InferencePrecision(self._precision),
            grid=grid,
        )

        return writer

    def _get_writer_data_args(
        self, pl_module: AhCoreLightningModule, data_format: DataFormat, stage: str
    ) -> tuple[tuple[int, int], float, tuple[int, int], tuple[int, int], int, Grid | None]:
        current_dataset: TiledWsiDataset
        current_dataset, _ = self._total_dataset.index_to_dataset(self._dataset_index)  # type: ignore
        slide_image = current_dataset.slide_image
        num_samples = len(current_dataset)

        if data_format == DataFormat.IMAGE or data_format == DataFormat.COMPRESSED_IMAGE:
            data_description: DataDescription = pl_module.data_description
            if data_description.inference_grid is None:
                raise ValueError("Inference grid is not defined in the data description.")
            inference_grid: GridDescription = data_description.inference_grid

            mpp = inference_grid.mpp
            if mpp is None:
                mpp = slide_image.mpp

            _, size = slide_image.get_scaled_slide_bounds(slide_image.get_scaling(mpp))

            # Let's get the data_description, so we can figure out the tile size and things like that
            tile_size = inference_grid.tile_size
            tile_overlap = inference_grid.tile_overlap

            if stage == "validate":
                grid = current_dataset._grids[0][0]  # pylint: disable=protected-access
            else:
                grid = None  # During inference we don't have a grid around ROI

        elif data_format == DataFormat.FEATURE:
            size = (num_samples, 1)
            mpp = 1.0
            tile_size = (1, 1)
            tile_overlap = (0, 0)
            num_samples = num_samples
            grid = current_dataset._grids[0][0]  # give grid, bc doesn't work otherwise

        else:
            raise NotImplementedError(f"Data format {data_format} is not yet supported.")

        return size, mpp, tile_size, tile_overlap, num_samples, grid
