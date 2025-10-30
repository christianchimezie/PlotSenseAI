# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Read README with explicit UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plotsense",
    version="0.1.3",
    author="Christian Chimezie, Toluwaleke Ogidan, Grace Farayola, Amaka Iduwe, Nelson Ogbeide, Onyekachukwu Ojumah, Olamilekan Ajao",
    author_email="chimeziechristiancc@gmail.com, gbemilekeogidan@gmail.com, gracefarayola@gmail.com, nwaamaka_iduwe@yahoo.com, Ogbeide331@gmail.com, Onyekaojumah22@gmail.com, olamilekan011@gmail.com",
    description="An intelligent plotting package with suggestions and explanations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/christianchimezie/PlotSenseAI",
    project_urls={
        "Documentation": "https://github.com/christianchimezie/PlotSenseAI/blob/main/README.md",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.8.0",
        "seaborn>=0.11",
        "pandas>=1.0",
        "numpy>=1.18",
        "python-dotenv",
        "groq",
        "anthropic",
        "openai",
        "google-genai",
        "requests",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-mock>=3.10",
            "flake8>=6.0",
            "autopep8>=2.0",
        ],
        "test": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-mock>=3.10",
        ],
    },
    license="Apache License 2.0",
)
