
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
