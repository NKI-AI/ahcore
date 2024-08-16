from typing import Any

import pyvips
from dlup.backends.common import AbstractSlideBackend
from dlup.types import PathLike

from ahcore.readers import StitchingMode, ZarrFileImageReader, H5FileImageReader

from enum import Enum
from typing import Any, Callable

from dlup.backends.openslide_backend import OpenSlideSlide
from dlup.backends.tifffile_backend import TifffileSlide
from dlup.backends.pyvips_backend import PyVipsSlide
from dlup.types import PathLike



class ImageBackend(Enum):
    """Available image backends."""

    OPENSLIDE: Callable[[PathLike], OpenSlideSlide] = OpenSlideSlide
    PYVIPS: Callable[[PathLike], PyVipsSlide] = PyVipsSlide
    TIFFFILE: Callable[[PathLike], TifffileSlide] = TifffileSlide
    H5: Callable[[PathLike], H5Slide] = H5Slide
    ZARR: Callable[[PathLike], ZarrSlide] = ZarrSlide

    def __call__(self, *args: "ImageBackend" | str) -> Any:
        return self.value(*args)


class ZarrSlide(AbstractSlideBackend):
    def __init__(self, filename: PathLike, stitching_mode: StitchingMode | str = StitchingMode.CROP) -> None:
        super().__init__(filename)
        self._reader: ZarrFileImageReader = ZarrFileImageReader(filename, stitching_mode=stitching_mode)
        self._spacings = [(self._reader.mpp, self._reader.mpp)]

    @property
    def size(self):
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
    def magnification(self):
        return None

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]) -> pyvips.Image:
        return self._reader.read_region(coordinates, level, size)

    def close(self):
        self._reader.close()


class H5Slide(AbstractSlideBackend):
    def __init__(self, filename: PathLike, stitching_mode: StitchingMode | str = StitchingMode.CROP) -> None:
        super().__init__(filename)
        self._reader: H5FileImageReader = H5FileImageReader(filename, stitching_mode=stitching_mode)
        self._spacings = [(self._reader.mpp, self._reader.mpp)]

    @property
    def size(self):
        return self._reader.size

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return (self._reader.size,)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return (1.0,)

    @property
    def vendor(self) -> str:
        return "H5FileImageReader"

    @property
    def properties(self) -> dict[str, Any]:
        return self._reader.metadata

    @property
    def magnification(self):
        return None

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]) -> pyvips.Image:
        return self._reader.read_region(coordinates, level, size)

    def close(self):
        self._reader.close()
