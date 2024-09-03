from typing import Any

import pyvips
from dlup.backends.common import AbstractSlideBackend
from dlup.types import PathLike  # type: ignore

from ahcore.readers import StitchingMode, ZarrFileImageReader


class ZarrSlide(AbstractSlideBackend):
    def __init__(self, filename: PathLike, stitching_mode: StitchingMode | str = StitchingMode.CROP) -> None:
        super().__init__(filename)
        self._reader: ZarrFileImageReader = ZarrFileImageReader(filename, stitching_mode=stitching_mode)
        self._spacings = [(self._reader.mpp, self._reader.mpp)]

    @property
    def size(self) -> tuple[int, int]:
        return self._reader.size

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return (self._reader.size,)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return (1.0,)

    @property
    def vendor(self) -> str:
        return "ZarrFileImageReader"

    @property
    def properties(self) -> dict[str, Any]:
        return self._reader.metadata

    @property
    def magnification(self) -> float | None:
        return None

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]) -> pyvips.Image:
        return self._reader.read_region(coordinates, level, size)

    def close(self) -> None:
        self._reader.close()
