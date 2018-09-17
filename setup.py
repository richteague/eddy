"""
Setup file for package `eddy`.
"""
from setuptools import find_packages, setup
import os


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setup(
    name="eddy",
    version="0.0.1",
    author="Richard Teague",
    author_email="rteague@umich.edu",
    py_modules=["eddy"],
    description=("Tools to recover expectionally precise rotation curves from "
                 "spatially resolved spectra."),
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/richteague/eddy",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['scipy', 'numpy', 'matplotlib', 'celerite', 'emcee'],
    zip_safe=False
)
