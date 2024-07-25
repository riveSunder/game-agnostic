from os.path import join, dirname, realpath
from setuptools import setup
import sys

setup(
    name="gameplr",
    py_modules=["gameplr"],
    version='0.1',
    install_requires=["numpy", "gym", "torch==2.2.0"],
    description="Game agnostic game player",
    author="Rive Sunder",
)
