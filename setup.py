# setup.py

"""
D-Scope Blink: C++ Extension Build Script
=========================================
This script compiles the C++ source files (e.g., dscope_accelerator.cpp)
into a Python extension module using pybind11 and setuptools. This allows
Python code to call the high-performance C++ functions.

Usage:
1. Make sure you have a C++ compiler, pybind11, and OpenCV installed.
   - On Linux (Ubuntu):
     sudo apt-get install build-essential python3-dev
     pip install pybind11
     sudo apt-get install libopencv-dev python3-opencv
   - On Windows:
     Install Visual Studio with C++ development tools.
     pip install pybind11 opencv-python
     (You may need to set environment variables for OpenCV headers/libs).

2. Run the build command from your terminal in the same directory:
   python setup.py install

This will create a compiled library (.so on Linux, .pyd on Windows) and
install it into your Python environment, making `import dscope_accelerator` possible.
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess

# --- pybind11 and OpenCV Configuration ---

class get_pybind_include(object):
    """Helper class to fetch the pybind11 include path."""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

def find_opencv_libs():
    """
    Tries to find OpenCV libraries and headers using pkg-config.
    This is the most reliable method on Linux. For Windows, manual
    paths or vcpkg might be necessary if this fails.
    """
    try:
        # Use pkg-config to get compiler and linker flags for OpenCV
        cflags = subprocess.check_output(['pkg-config', '--cflags', 'opencv4']).decode('utf-8').strip().split()
        ldflags = subprocess.check_output(['pkg-config', '--libs', 'opencv4']).decode('utf-8').strip().split()
        
        include_dirs = [flag[2:] for flag in cflags if flag.startswith('-I')]
        library_dirs = [flag[2:] for flag in ldflags if flag.startswith('-L')]
        libraries = [flag[2:] for flag in ldflags if flag.startswith('-l')]
        
        print(f"Found OpenCV includes: {include_dirs}")
        print(f"Found OpenCV lib dirs: {library_dirs}")
        print(f"Found OpenCV libs: {libraries}")
        
        return include_dirs, library_dirs, libraries
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: pkg-config for opencv4 not found.", file=sys.stderr)
        print("Attempting fallback for common installations.", file=sys.stderr)
        
        # Fallback for Windows or systems without pkg-config
        if sys.platform == 'win32':
            # This requires OpenCV to be installed in a standard location
            # or for OPENCV_DIR to be set as an environment variable.
            opencv_dir = os.environ.get('OPENCV_DIR')
            if opencv_dir:
                include_dirs = [os.path.join(opencv_dir, 'build', 'include')]
                library_dirs = [os.path.join(opencv_dir, 'build', 'x64', 'vc15', 'lib')] # Adjust for your VS version
                libraries = ['opencv_world455'] # Adjust for your OpenCV version
                return include_dirs, library_dirs, libraries
            else:
                 print("ERROR: On Windows, please set the OPENCV_DIR environment variable.", file=sys.stderr)
        
        # Fallback for Linux if pkg-config is missing but headers are in standard locations
        elif sys.platform.startswith('linux'):
             include_dirs = ['/usr/include/opencv4']
             library_dirs = []
             libraries = ['opencv_core', 'opencv_imgproc', 'opencv_highgui', 'opencv_imgcodecs']
             if os.path.exists('/usr/include/opencv4'):
                 return include_dirs, library_dirs, libraries

        print("ERROR: Could not automatically determine OpenCV paths.", file=sys.stderr)
        return [], [], []


# Get OpenCV paths
opencv_include_dirs, opencv_library_dirs, opencv_libraries = find_opencv_libs()

# Define the C++ extension module
ext_modules = [
    Extension(
        'dscope_accelerator', # Name of the Python module
        ['dscope_accelerator.cpp'], # List of C++ source files
        include_dirs=[
            get_pybind_include(),
            *opencv_include_dirs
        ],
        library_dirs=opencv_library_dirs,
        libraries=opencv_libraries,
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-Wall'] if sys.platform != 'win32' else ['/std:c++17', '/O2']
    ),
]

setup(
    name='dscope_accelerator',
    version='0.1.0',
    author='Gemini',
    author_email='na@example.com',
    description='C++ accelerator for D-Scope Blink image processing.',
    long_description='Provides high-performance C++ implementations for key functions.',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
)
