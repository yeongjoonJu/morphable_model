"""
Used to compile the Cython code
"""
#from setuptools import setup, find_packages
import numpy
from distutils.core import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# ext = Extension(name="hello", sources=["hello.pyx"])
# setup(ext_modules=cythonize(ext))

ext_modules = [
    Extension("mesh_utils",
        sources = ["mesh_utils.pyx", "mesh_core.cpp"],
        language='c++', include_dirs=[numpy.get_include()]
    )
]

setup(
    name = "morphable",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)