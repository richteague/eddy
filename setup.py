#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="eddy",
    version="1.1",
    author="Richard Teague",
    author_email="rteague@umich.edu",
    description=("Tools to recover expectionally precise rotation curves from "
                 "spatially resolved spectra."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richteague/eddy",
    packages=["eddy"],
    license="MIT",
    install_requires=["scipy", "numpy", "matplotlib", "emcee", "corner"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
