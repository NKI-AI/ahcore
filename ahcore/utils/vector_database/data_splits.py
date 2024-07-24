import json
import os
from pathlib import Path

import numpy as np
from dlup.annotations import WsiAnnotations
from skmultilearn.model_selection import IterativeStratification

from ahcore.utils.vector_database.utils import calculate_total_annotation_area


def save_to_json(train_filenames, val_filenames, test_filenames, file_path):
    """
    Saves the lists of filenames to a JSON file.

    :param train_filenames: List of training filenames
    :param val_filenames: List of validation filenames
    :param test_filenames: List of test filenames
    :param file_path: Path to the JSON file where data will be saved
    """
    data = {"train": train_filenames, "validation": val_filenames, "test": test_filenames}

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def load_from_json(file_path):
    """
    Loads the lists of filenames from a JSON file.

    :param file_path: Path to the JSON file from which data will be loaded
    :return: A tuple containing three lists of filenames (train, validation, test)
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["train"], data["validation"], data["test"]


def find_files_with_extension(path: str, extension: str) -> list[str]:
    """
    Find all files in a directory with a specific extension.
    """
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def find_annotation_area_per_class(path: str, classes: list[str]) -> dict[str, float]:
    scaling = 1.0
    area_dict = {}
    for class_name in classes:
        annotation = WsiAnnotations.from_geojson(path)
        annotation.filter(class_name)
        area = calculate_total_annotation_area(annotation, scaling=scaling)
        area_dict[class_name] = area
    return area_dict


def find_annotation_area_per_class_in_directory(annotation_dir: str, classes: list[str]) -> dict[str, dict[str, float]]:
    annotation_files = find_files_with_extension(annotation_dir, ".geojson")
    annotation_area_dict = {}
    for annotation_file in annotation_files:
        filename = Path(annotation_file).stem
        area_dict = find_annotation_area_per_class(annotation_file, classes)
        annotation_area_dict[filename] = area_dict
    return annotation_area_dict


def print_class_distribution(data: dict[str, dict[str, float]], filenames: list[str]):
    """
    Prints the class distribution as percentages for a given subset of data.

    :param data: Complete dictionary of data.
    :param filenames: Filenames corresponding to the subset for which distribution is calculated.
    """
    total_areas = {key: 0.0 for key in next(iter(data.values())).keys()}
    number_of_nonzero_files = {}
    subset_total = 0

    # Accumulate areas for the specified filenames
    for filename in filenames:
        for key, area in data[filename].items():
            total_areas[key] += area
            subset_total += area
            if area > 0:
                if key not in number_of_nonzero_files:
                    number_of_nonzero_files[key] = 0
                number_of_nonzero_files[key] += 1

    # Print the distribution
    if subset_total > 0:
        print({key: 100 * area / subset_total for key, area in total_areas.items()})
    else:
        print({key: 0 for key in total_areas.keys()})
    print(f"Number of files with non-zero area per class: {number_of_nonzero_files}")
    print(total_areas)


def stratified_split(
    data: dict[str, dict[str, float]], split_ratios: list[float], seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """
    Stratifies data into train, validation, and test sets based on individual class areas,
    using a two-step iterative stratification.

    :param data: Dictionary with filenames as keys and dictionaries of class areas as values.
    :param split_ratios: List of three floats indicating the proportion of train, validation, and test sets.
    :param seed: Random seed for reproducibility.
    """
    np.random.seed(seed)  # Ensure reproducibility
    # Generate a list of labels and a matrix where each row corresponds to a file and columns to class areas
    classes = sorted(next(iter(data.values())).keys())
    filenames = list(data.keys())

    to_be_removed = ["K109373-1.svs", "K109373-2.svs"]  # Manual remove these files since no corresponding slide exists
    filenames = [filename for filename in filenames if filename not in to_be_removed]

    labels_matrix = np.array([[data[filename][cls] for cls in classes] for filename in filenames])

    # Filter out empty files
    labels_sum = labels_matrix.sum(axis=1)
    valid_indices = labels_sum > 0
    labels_matrix = labels_matrix[valid_indices]
    filenames = [filename for filename, valid in zip(filenames, valid_indices) if valid]

    # Prepare the first stratification to separate initial train and temp (val+test)
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[split_ratios[2] + split_ratios[1], split_ratios[0]]
    )
    train_indexes, temp_indexes = next(stratifier.split(labels_matrix, labels_matrix))

    # Split temp into validation and test using the adjusted ratios
    temp_labels_matrix = labels_matrix[temp_indexes]
    temp_filenames = [filenames[idx] for idx in temp_indexes]
    val_percentage = split_ratios[1] / (split_ratios[1] + split_ratios[2])
    stratifier_temp = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1 - val_percentage, val_percentage]
    )
    val_indexes, test_indexes = next(stratifier_temp.split(temp_labels_matrix, temp_labels_matrix))

    # Retrieve filenames for each set
    train_filenames = [filenames[idx] for idx in train_indexes]
    val_filenames = [temp_filenames[idx] for idx in val_indexes]
    test_filenames = [temp_filenames[idx] for idx in test_indexes]

    # Print class distributions
    print("Train set class distribution:")
    print_class_distribution(data, train_filenames)
    print("Validation set class distribution:")
    print_class_distribution(data, val_filenames)
    print("Test set class distribution:")
    print_class_distribution(data, test_filenames)

    return train_filenames, val_filenames, test_filenames


if __name__ == "__main__":
    annotation_dir = "/path/to/annotation/dir"
    classes = ["T", "D", "ND", "I", "UT", "IE", "MS"]
    annotation_area_dict = find_annotation_area_per_class_in_directory(annotation_dir, classes)
    train_list, val_list, test_list = stratified_split(annotation_area_dict, [0.62, 0.19, 0.19], seed=1)
