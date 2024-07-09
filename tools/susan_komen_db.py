from pathlib import Path

from dlup import SlideImage
from dlup.backends import ImageBackend  # type: ignore

from ahcore.utils.database_models import (
    CategoryEnum,
    Image,
    Manifest,
    Mask,
    Patient,
    Split,
    SplitDefinitions,
)
from ahcore.utils.manifest import open_db


def get_patient_from_sk_id(filename: str) -> str:
    return filename[-10:-4]


def populate_from_annotated_sk(
    session,
    image_folder: Path,
    mask_folder: Path,
):
    """This is a basic example, adjust to your needs."""
    # TODO: We should do the mpp as well here

    manifest = Manifest(name="v20240708vDEBUG")
    session.add(manifest)
    session.flush()

    split_definition = SplitDefinitions(version="v1", description="Initial split")
    session.add(split_definition)
    session.flush()

    for iter, file in enumerate(image_folder.glob("*.svs")):
        #-rw-r--r-- 1 j.teuwen j.teuwen  86K Jan 12 16:55  K102483.svs.geojson
        #-rw-r--r-- 1 j.teuwen j.teuwen 112K Jan 12 16:55  K102487.svs.geojson
        #-rw-r--r-- 1 j.teuwen j.teuwen 133K Jan 12 16:55  K102489.svs.geojson
        #-rw-r--r-- 1 j.teuwen j.teuwen  40K Jan 12 16:56  K102490.svs.geojson
        #-rw-r--r-- 1 j.teuwen j.teuwen  83K Jan 12 16:56  K102491.svs.geojson
        if file.name not in ["K102483.svs", "K102487.svs", "K102489.svs", "K102490.svs", "K102491.svs"]:
            continue
        patient_code = get_patient_from_sk_id(file.name)

        mask_path = mask_folder / file.name.replace(".svs", ".tiff")

        # Only add patient if it doesn't exist
        existing_patient = session.query(Patient).filter_by(patient_code=patient_code).first()  # type: ignore
        if existing_patient:
            patient = existing_patient
        else:
            patient = Patient(patient_code=patient_code, manifest=manifest)
            session.add(patient)
            session.flush()

            # For now random.
            split_category = CategoryEnum.PREDICT
            split = Split(
                category=split_category,
                patient=patient,
                split_definition=split_definition,
            )
            session.add(split)
            session.flush()

        with SlideImage.from_file_path(
            image_folder / file.name, backend=ImageBackend.PYVIPS # type: ignore
        ) as slide:  # type: ignore
            mpp = slide.mpp
            width, height = slide.size
            image = Image(
                filename=str(file.name),
                mpp=mpp,
                height=height,
                width=width,
                reader="PYVIPS",
                patient=patient,
            )
        print(mpp)
        session.add(image)
        session.flush()  # Flush so that Image ID is populated for future records

        mask = Mask(filename=str(mask_path), reader="TIFFFILE", image=image)
        session.add(mask)
        session.commit()


if __name__ == "__main__":
    annotation_folder = Path("SusanKomen/")
    image_folder = Path("/data/groups/aiforoncology/archive/pathology/SusanKomen/images/")
    with open_db("sqlite:////home/e.marcus/projects/ahcore/testdb/debug.db", False) as session:
        populate_from_annotated_sk(session, image_folder, annotation_folder)
