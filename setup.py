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


with open("README.rst") as readme_file:
    long_description = readme_file.read()

install_requires = [
    "numpy>=1.25.2",
    "torch>=2.2.2",
    "pillow>=10.2.0",
    "pytorch-lightning>=2.2.2",
    "torchvision>=0.17.2",
    "pydantic>=2.6.4",
    "tensorboard>=2.14.0",
    "mlflow>=2.6.0",
    "hydra-core>=1.3.2",
    "python-dotenv>=1.0.0",
    "tqdm>=4.64",
    "rich>=12.4",
    "hydra-submitit-launcher>=1.2.0",
    "hydra-optuna-sweeper>=1.3.0.dev0",
    "hydra-colorlog>=1.2.0",
    "dlup>=0.3.38",
    "kornia>=0.7.2",
    "h5py>=3.8.0",
    "monai[einops]>=1.3.0",
    "imagecodecs==2024.1.1",
    "zarr==2.17.2",
    "sqlalchemy>=2.0.21",
    "imageio>=2.34.0",
]


setup(
    author="AI for Oncology Lab @ The Netherlands Cancer Institute",
    long_description=long_description,
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
            "mypy==1.9.0",
            "black==23.7.0",
            "types-Pillow",
            "sphinx",
            "sphinx_copybutton",
            "numpydoc",
            "myst-parser",
            "sphinx-book-theme",
            "pre-commit",
            "tox",
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
