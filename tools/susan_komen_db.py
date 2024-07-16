from pathlib import Path

from dlup import SlideImage
from dlup.backends import ImageBackend  # type: ignore

from ahcore.utils.database_models import (
    CategoryEnum,
    Image,
    ImageAnnotations,
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
    annotation_folder: Path,
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
        if file.name not in ["K102483.svs", "K102487.svs", "K102489.svs", "K102490.svs", "K102491.svs"]:
            continue
        patient_code = get_patient_from_sk_id(file.name)

        mask_path = mask_folder / file.name.replace(".svs", ".tiff")
        annotation_path = annotation_folder / file.name.replace(".svs", ".svs.geojson")

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
            image_folder / file.name, backend=ImageBackend.PYVIPS  # type: ignore
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

        mask = Mask(filename=str(mask_path.name), reader="TIFFFILE", image=image)
        session.add(mask)
        session.commit()

        # check if annotation exists, if so add it
        if annotation_path.exists():
            image_annotation = ImageAnnotations(filename=str(annotation_path.name), reader="GEOJSON", image=image)
            session.add(image_annotation)
            session.commit()


if __name__ == "__main__":
    mask_folder = Path("/processing/e.marcus/susankomen_data/masks/SusanKomen")
    annotation_folder = Path("/processing/e.marcus/susankomen_data/annotations/v20250625/")
    image_folder = Path("/processing/e.marcus/susankomen_data/images/")
    with open_db("sqlite:////home/e.marcus/projects/ahcore/testdb/debug.db", False) as session:
        populate_from_annotated_sk(session, image_folder, mask_folder, annotation_folder)
