#!/usr/bin/python
# -*- coding: utf8 -*-
#
#   bem: triangulation and fmm/bem electrostatics tools 
#
#   Copyright (C) 2011-2012 Robert Jordens <jordens@gmail.com>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Hack to prevent stupid error on exit of `python setup.py test`. (See
# http://www.eby-sarna.com/pipermail/peak/2010-May/003357.html.)
# https://github.com/erikrose/more-itertools/commit/da7e3c771523711adeaef3c6a67ba99de5e2e81a


try:
    from setuptools import setup, Extension, find_packages
except ImportError:
    from distutils import setup
    from distutils.extension import Extension

import numpy
import Cython
from Cython.Distutils import build_ext
import sys
import os
from os.path import join as pjoin
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from icecream import ic
import numpy as np

plat = sys.platform
print('operating system:',sys)

# for Windows System
if plat == 'win32':
    tri_dir = "triangle-win"
# for Linux or mac System
elif plat == 'darwin' or 'linux' or 'linux2':
    tri_dir = "triangle"
else:
    raise TypeError('operating system not found')

def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted from http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """
    Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()
# ic(CUDA)


setup(
    name="bem",
    description="BEM FMM Laplace solver",
    long_description= """Python bindings for Fastlap""",
    version="1.0+dev",
    author="Robert Jordens, Ben Saarel, Shuqi Xu, Qian Yu, Nicole Greene, et. al.",
    url="https://github.com/Andris-Huang/bem",
    license="multiple",
    # python_requires="<=3.10",
    install_requires=[
                    "numpy", 
                    "pandas",
                    "cython",
                    "jupyter",
                    "jupyterlab",
                    "scipy",
                    "matplotlib",
                    "cvxopt",
                    "cvxpy",
                    "apptools",
                    "envisage",
                    "ipyparallel",
                    "pyvista"
                    ],
    packages = find_packages(),
    test_suite = "bem.tests",
    cmdclass = {"build_ext": build_ext},
    ext_modules = [
        Extension("bem.fastlap",
            define_macros = [],
            extra_compile_args=["-ffast-math"],
            sources = ["bem/fastlap.pyx",],
            libraries=['cudart', 'cublas'],
            language='c++',
            library_dirs=[CUDA['lib64']],
            include_dirs = [
                CUDA['include'],
                "fastlap_cuda",
                numpy.get_include(),],
            extra_objects=["lib/fastlap.a"],
            # sources = [
            #     "bem/fastlap.pyx",
            #     "bem/fastlap_support.c",
            #     "fastlap/fastlap.c",
            #     "fastlap/calcp.c",
            #     "fastlap/direct.c",
            #     "fastlap/memtracker.c",
            #     "fastlap/mulDisplay.c",
            #     "fastlap/mulDo.c",
            #     "fastlap/mulMats.c",
            #     "fastlap/mulGlobal.c",
            #     "fastlap/mulMulti.c",
            #     "fastlap/mulLocal.c",
            #     "fastlap/mulSetup.c",],
            # include_dirs = [
            #     "fastlap",
            #     numpy.get_include(),],
        ),
        Extension("bem.pytriangle",
            define_macros = [
                ("TRILIBRARY", "1"),
                ("NO_TIMER", "1"),
                # ("REDUCED", "1"),
                ("REAL", "double"),
                ("EXTERNAL_TEST", "1"),],
            # extra_compile_args=["-ffast-math"],
            sources = [
                "bem/pytriangle.pyx",
                tri_dir+"/triangle.c",],
            include_dirs = [
                tri_dir,
                numpy.get_include(),],
        ),
    ],
)
