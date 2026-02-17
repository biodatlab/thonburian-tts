#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name="thonburian-tts",
    version="1.0.0",
    author="Looloo Technology, WordSense, Biomedical and Data Lab",
    author_email="",
    description="Thai Text-to-Speech (TTS) engine built on top of F5-TTS with voice cloning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biodatlab/thonburian-tts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
        "gradio": [
            "gradio",
        ],
    },
    entry_points={
        "console_scripts": [
            "thonburian-tts=flowtts.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "flowtts": [
            "configs/**/*.yaml",
            "configs/**/*.yml",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/biodatlab/thonburian-tts/issues",
        "Source": "https://github.com/biodatlab/thonburian-tts",
        "Documentation": "https://github.com/biodatlab/thonburian-tts/blob/main/README.md",
        "Paper": "https://ieeexplore.ieee.org/document/11320472",
        "Model Checkpoints": "https://huggingface.co/biodatlab/ThonburianTTS",
    },
    keywords=[
        "text-to-speech",
        "tts",
        "thai",
        "voice-cloning",
        "f5-tts",
        "flow-matching",
        "neural-speech-synthesis",
        "ai",
        "machine-learning",
        "deep-learning",
    ],
)