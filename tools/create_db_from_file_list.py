import argparse
import datetime
from pathlib import Path
from rich.progress import Progress

from dlup import SlideImage
from dlup.experimental_backends import ImageBackend  # type: ignore

from ahcore.utils.database_models import CategoryEnum, Image, Manifest, Patient, Split, SplitDefinitions
from ahcore.utils.io import get_logger
from ahcore.utils.manifest import open_db

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="This script will create an inference database based on a file list.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input image path. Can be used for a single file. "
        "In case of multiple files, provide a list to --input-list",
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        help="Input file path. Can be used for a list of files, where each line is a file path. "
        "In case of a single file, provide the path to --input",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output sqlite database name",
        required=True,
    )
    args = parser.parse_args()

    if args.input and args.input_list:
        raise ValueError("Only one of --input or --input-list should be provided")

    if not args.input and not args.input_list:
        raise ValueError("Either --input or --input-list should be provided")

    if not args.input_list:
        files_list = [args.input]

    else:
        with open(args.input_list, "r") as f:
            files_list = f.readlines()
            files_list = [Path(f.strip()) for f in files_list if f.strip()]

    # Verify if the files exist and drop if they are not with a logger warning
    for file in files_list:
        if not file.exists():
            logger.warning(f"{file} does not exist. Dropping from the list.")
            files_list.remove(file)

    if not files_list:
        logger.warning("No files found. Exiting.")
        return

    logger.info(f"Creating database {args.output}")

    if args.output.exists():
        logger.warning(f"Output file {args.output} already exists. Exiting.")
        return

    with open_db(f"sqlite:///{args.output}", ensure_exists=False) as session:
        # Lets create a version name based on the current date in the form of vYYYYMMDD
        version_name = datetime.datetime.now().strftime("v%Y%m%d-ahcore-inference")
        manifest = Manifest(name=version_name)
        session.add(manifest)
        session.flush()

        split_definition = SplitDefinitions(version="inference-split", description="inference-split")
        session.add(split_definition)
        session.flush()

        with Progress() as progress:
            task = progress.add_task("[cyan]Creating...", total=len(files_list))
            for filename in files_list:
                patient_id = str(filename)  # TODO: This shouldn't be needed for inference
                patient = Patient(patient_code=patient_id, manifest=manifest)
                session.add(patient)
                session.flush()

                split = Split(
                    category=CategoryEnum.PREDICT,
                    patient=patient,
                    split_definition=split_definition,
                )
                session.add(split)
                session.flush()

                try:
                    with SlideImage.from_file_path(filename, backend=ImageBackend.OPENSLIDE) as slide:
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

                except Exception as e:
                    logger.warning(f"Failed to read {filename} with OPENSLIDE: {e}. Skipping.")
                    continue

                progress.update(task, advance=1)

        session.commit()


if __name__ == "__main__":
    main()
