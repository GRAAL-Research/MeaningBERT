import os

from setuptools import setup, find_packages

current_file_path = os.path.abspath(os.path.dirname(__file__))


def get_readme():
    readme_file_path = os.path.join(current_file_path, "README.md")
    with open(readme_file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_version():
    version_file_path = os.path.join(current_file_path, "version.txt")
    with open(version_file_path, "r", encoding="utf-8") as f:
        version = f.read().strip()
    return version


setup(
    name="meaningbert",
    version=get_version(),
    author="David Beauchemin",
    author_email="david.beauchemin@ift.ulaval.ca",
    url="https://github.com/GRAAL-Research/MeaningBERT",
    license="MIT",
    description="MeaningBERT: assessing meaning preservation between sentences",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.6.0",
        "transformers",
        "datasets",
        "evaluate",
        "numpy",
        "scipy",
        "scikit-learn",
        "poutyne",
    ],
    extras_require={
        "training": [
            "sentence-transformers",
            "textstat",
            "tqdm",
            "sacremoses",
            "rake_nltk",
            "rouge_score",
            "wandb",
            "accelerate",
        ],
        "all": [
            "sentence-transformers",
            "textstat",
            "tqdm",
            "sacremoses",
            "rake_nltk",
            "rouge_score",
            "wandb",
            "accelerate",
            "pytest",
            "pytest_cov",
            "black",
            "pylint",
            "pre-commit",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
