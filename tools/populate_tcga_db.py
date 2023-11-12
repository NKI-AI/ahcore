"""This is an example on how to populate an ahcore manifest database using the TCGA dataset."""
import json
import random
from pathlib import Path

from dlup import SlideImage
from dlup.experimental_backends import ImageBackend  # type: ignore

from ahcore.utils.database_models import (
    CategoryEnum,
    Image,
    ImageAnnotations,
    ImageLabels,
    Manifest,
    Mask,
    Patient,
    PatientLabels,
    Split,
    SplitDefinitions,
)
from ahcore.utils.manifest import open_db


def get_patient_from_tcga_id(tcga_filename: str) -> str:
    return tcga_filename[:12]


def populate_from_annotated_tcga(
    session,
    image_folder: Path,
    annotation_folder: Path,
    path_to_mapping: Path,
    predict: bool = False,
):
    """This is a basic example, adjust to your needs."""
    # TODO: We should do the mpp as well here

    with open(path_to_mapping, "r") as f:
        mapping = json.load(f)
    manifest = Manifest(name="v20230228")
    session.add(manifest)
    session.flush()

    split_definition = SplitDefinitions(version="v1", description="Initial split")
    session.add(split_definition)
    session.flush()

    for folder in annotation_folder.glob("TCGA*"):
        patient_code = get_patient_from_tcga_id(folder.name)

        if not predict:
            annotation_path = folder / "annotations.json"
            mask_path = folder / "roi.json"

        # Only add patient if it doesn't exist
        existing_patient = session.query(Patient).filter_by(patient_code=patient_code).first()  # type: ignore
        if existing_patient:
            patient = existing_patient
        else:
            patient = Patient(patient_code=patient_code, manifest=manifest)
            session.add(patient)
            session.flush()

            # For now random.
            if predict:
                split_category = CategoryEnum.PREDICT
            else:
                split_category = random.choices(
                    [CategoryEnum.FIT, CategoryEnum.VALIDATE, CategoryEnum.TEST],
                    [67, 33, 0],
                )[0]

            split = Split(
                category=split_category,
                patient=patient,
                split_definition=split_definition,
            )
            session.add(split)
            session.flush()

        # Add only the label if it does not exist yet.
        existing_label = session.query(PatientLabels).filter_by(key="study", patient_id=patient.id).first()
        if not existing_label:
            patient_label = PatientLabels(key="study", value="BRCA", patient=patient)
            session.add(patient_label)
            session.flush()

        filename = mapping[folder.name]

        # TODO: OPENSLIDE doesn't work
        kwargs = {}
        if (
            "TCGA-OL-A5RY-01Z-00-DX1.AE4E9D74-FC1C-4C1E-AE6D-5DF38899BBA6.svs" in filename
            or "TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.svs" in filename
        ):
            kwargs["overwrite_mpp"] = (0.25, 0.25)

        with SlideImage.from_file_path(
            image_folder / filename, backend=ImageBackend.PYVIPS, **kwargs  # type: ignore
        ) as slide:  # type: ignore
            mpp = slide.mpp
            width, height = slide.size
            image = Image(
                filename=str(filename),
                mpp=mpp,
                height=height,
                width=width,
                reader="OPENSLIDE",
                patient=patient,
            )
        session.add(image)
        session.flush()  # Flush so that Image ID is populated for future records

        if not predict:
            mask = Mask(filename=str(mask_path), reader="GEOJSON", image=image)
            session.add(mask)

            image_annotation = ImageAnnotations(filename=str(annotation_path), reader="GEOJSON", image=image)
            session.add(image_annotation)

        # Randomly decide if it's cancer or benign
        image_label = ImageLabels(
            key="tumor_type", value="cancer" if random.choice([True, False]) else "benign", image=image
        )
        session.add(image_label)

        session.commit()


if __name__ == "__main__":
    annotation_folder = Path("tissue_subtypes/v20230228_debug/")
    image_folder = Path("/data/groups/aiforoncology/archive/pathology/TCGA/images/")
    path_to_mapping = Path("/data/groups/aiforoncology/archive/pathology/TCGA/identifier_mapping.json")
    with open_db("sqlite:///manifest.db") as session:
        populate_from_annotated_tcga(session, image_folder, annotation_folder, path_to_mapping, predict=True)
