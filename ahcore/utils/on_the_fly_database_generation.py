"""
Functions for generating an in-memory MinimalImage database on-the-fly with only an image root directory and glob
pattern. Used for inference of, e.g., a segmentation model on a directory filled with WSIs, without generating a
database explicitly.
"""

from pathlib import Path

import sqlalchemy
from dlup import SlideImage
from dlup.experimental_backends import ImageBackend
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ahcore.utils.database_models import MinimalImage, OnTheFlyBase
from ahcore.utils.manifest import create_tables, open_db_from_engine


def populate_from_directory_and_glob_pattern(session: Session, image_folder: Path | str, glob_pattern: str) -> None:
    """
    Populates the MinimalImage database in the in-memory session
        with slides found in the image_folder using the
        glob_pattern. Population happens in-place, so no return.

    Parameters
    ----------
    session : Session
        The opened session that connects to the DB engine
    image_folder : str
        The root directory of the images
    glob_pattern : str
        The glob pattern to find images within the root directory

    Returns
    -------
    None
    """
    for wsi in Path(image_folder).glob(glob_pattern):
        with SlideImage.from_file_path(
            image_folder / wsi,
            backend=ImageBackend.PYVIPS,  # type: ignore
        ) as slide:  # type: ignore
            mpp = slide.mpp
            width, height = slide.size
            image = MinimalImage(
                filename=str(wsi),
                mpp=mpp,
                height=height,
                width=width,
                reader="PYVIPS",
                relative_filename=str(wsi.relative_to(image_folder)),
            )
        session.add(image)
        session.flush()  # Flush so that Image ID is populated for future records
        session.commit


def get_populated_in_memory_db(image_folder: Path, glob_pattern: str) -> Engine:
    """
    Callable function to get the populated in-memory DB as an Engine

    Parameters
    ----------
    image_folder : str
        The root directory of the images
    glob_pattern : str
        The glob pattern to find images within the root directory

    Returns
    -------
    Engine
        an in-memory sqlalchemy Engine
    """
    assert image_folder.is_dir(), f"image_folder ({image_folder}) does not exist"

    assert (
        len([i for i in image_folder.glob(glob_pattern)]) > 0
    ), f"No images found in {image_folder} with glob pattern {glob_pattern}"

    # An empty URL will create a `:memory:` database
    engine = sqlalchemy.create_engine("sqlite://")
    create_tables(engine=engine, base=OnTheFlyBase)

    with open_db_from_engine(engine) as session:
        # Populate the DB through the session. Happens in-place
        populate_from_directory_and_glob_pattern(session, image_folder, glob_pattern)

        # Commit is required before passing the engine back. If not commited, the engine and
        # session here will contain the information, But outside of this context it will be lost.
        session.commit()

        # Return the engine object that is bound to the session, so we can close the session
        return engine
