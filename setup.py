#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:21:28 2019

@author: thosvarley
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("opyn.pyx", annotate = True),
    include_dirs=[numpy.get_include()]
)
