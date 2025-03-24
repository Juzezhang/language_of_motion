from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name="Language_of_motion",
    version="0.1.0",
    author="Changan, Juze, Shrinidhi",
    author_email="juze@stanford.edu",
    description="The Language of Motion: Unifying Verbal and Non-verbal Language of 3D Human Motion.",
    packages=find_packages(exclude=("configs", "deps")),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
    ],
)
