from setuptools import setup, Extension
import pybind11


#Compile with python setup.py build_ext --inplace
setup(
    ext_modules=[
        Extension(
            "hsne_wrapper",  # Module name
            ["hsne_cmd.cpp"],  # C++ source file
            include_dirs=[pybind11.get_include(), "/usr/include"],  # Include paths
            libraries=["lz4", "hdiutils", "hdidimensionalityreduction", "hdidata"],  # Link against LZ4
            library_dirs=[ "/home/jacob253/minconda3/envs/htsne/lib"],  # Library paths
            extra_compile_args=["-fopenmp"],
            extra_link_args=['-fopenmp'],
            language="c++",
        )
    ],
)
