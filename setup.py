#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:21:28 2019

@author: thosvarley
"""

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("OPyN", ["OPyN/opyn.pyx"], 
              include_dirs=[np.get_include()])
]

setup(
    name="OPyN",
    version="0.1",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
    packages=find_packages(exclude=("tutorial",))
)
