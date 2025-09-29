"""
Setup script for RL-STIR project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl-stir",
    version="0.1.0",
    author="RL-STIR Team",
    description="Reinforcement Learning for Security Threat Investigation and Response",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "pyarrow>=10.0.0",
        "scikit-learn>=1.3.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "gymnasium>=0.29.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
        "torchtyping",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "evtx": [
            "evtx>=0.8.0",
            "python-evtx>=0.8.0",
        ],
    },
)
