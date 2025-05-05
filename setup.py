# setup.py
from setuptools import setup, find_packages

setup(
    name="tjsm_gpu",
    version="0.1.0",
    package_dir={"": "code"},
    packages=find_packages(where="code"),
    install_requires=[
        "matplotlib==3.10.1",
        "numpy==2.2.5",
        "torch==2.6.0",
        "ipykernel"
    ],
    author="Chris",
    description="Your GPU-accelerated TJSM code",
)