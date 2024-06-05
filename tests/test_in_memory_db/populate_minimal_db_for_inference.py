"""
This is a test for in-memory on-the-fly generation of a minimal ahcore database using a dummy dataset
using tiny .svs files from openslide
"""

from pathlib import Path

from sqlalchemy.orm import sessionmaker

from ahcore.utils.database_models import MinimalImage
from ahcore.utils.on_the_fly_database_generation import get_populated_in_memory_db

if __name__ == "__main__":
    image_folder = Path(__file__) / "test_in_memory_db"
    glob_pattern = "**/*.svs"
    engine = get_populated_in_memory_db(image_folder=image_folder, glob_pattern=glob_pattern)

    with sessionmaker(bind=engine)() as session:
        table = session.query(MinimalImage).all()

    assert len(table) > 0, "The database was not populated"
    assert len(table) == 3, f"The database has {len(table)} entires while there are only 3 test images"
