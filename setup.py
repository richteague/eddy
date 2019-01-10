#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from distutils.core import setup

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setup(
    name="eddy",
    version="0.0.1",
    author="Richard Teague",
    author_email="rteague@umich.edu",
    packages=['eddy', 'eddy.tests'],
    url="https://github.com/richteague/eddy",
    description=("Tools to recover expectionally precise rotation curves from "
                 "spatially resolved spectra."),
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="MIT",
    include_package_data=True,
    install_requires=['scipy', 'numpy', 'matplotlib', 'emcee', 'celerite'],
    zip_safe=False
)
