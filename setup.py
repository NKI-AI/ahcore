#!/usr/bin/env python
# coding=utf-8
"""The setup script."""
import ast

from setuptools import find_packages, setup  # type: ignore  # noqa

with open("ahcore/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


with open("README.md") as readme_file:
    long_description = readme_file.read()

install_requires = [
    "numpy>=1.25.2",
    "torch>=2.0.1",
    "pillow>=9.5.0",
    "pytorch-lightning>=2.0.8",
    "torchvision>=0.15.2",
    "pydantic>=2.0.3",
    "tensorboard>=2.14.0",
    "mlflow>=2.6.0",
    "hydra-core>=1.3.2",
    "python-dotenv>=1.0.0",
    "tqdm>=4.64",
    "rich>=12.4",
    "hydra-submitit-launcher>=1.2.0",
    "hydra-optuna-sweeper>=1.3.0.dev0",
    "hydra-colorlog>=1.2.0",
    "dlup>=0.3.30",
    "kornia>=0.7.0",
    "h5py>=3.8.0",
    "monai[einops]==1.2.0",
    "imagecodecs==2023.9.4",
]


setup(
    author="AI for Oncology Lab @ The Netherlands Cancer Institute",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "ahcore=ahcore.cli:main",
        ],
    },
    description="Ahcore the AI for Oncology core components for computational pathology.",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest",
            "numpydoc",
            "pylint==2.17.7",
            "black==23.9.1",
            "types-Pillow",
            "sphinx",
            "sphinx_copybutton",
            "numpydoc",
            "myst-parser",
            "sphinx-book-theme",
        ],
    },
    license="Apache Software License 2.0",
    include_package_data=True,
    name="ahcore",
    test_suite="tests",
    url="https://github.com/NKI-AI/ahcore",
    py_modules=["ahcore"],
    version=version,
)
