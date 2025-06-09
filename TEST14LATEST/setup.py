# setup.py

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
    Tries to find OpenCV libraries and headers.
    Uses pkg-config on Linux and OPENCV_DIR environment variable on Windows.
    """
    # Linux method using pkg-config
    if sys.platform.startswith('linux'):
        try:
            print("Attempting to find OpenCV with pkg-config...")
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
            print("WARNING: pkg-config for opencv4 not found. Check your OpenCV installation.", file=sys.stderr)
            return [], [], []

    # Windows method using OPENCV_DIR
    elif sys.platform == 'win32':
        opencv_dir = os.environ.get('OPENCV_DIR')
        if not opencv_dir:
            print("ERROR: On Windows, please set the OPENCV_DIR environment variable.", file=sys.stderr)
            print("       This should point to your OpenCV 'build' directory (e.g., C:\\opencv\\build).", file=sys.stderr)
            return [], [], []

        print(f"Using OPENCV_DIR: {opencv_dir}")
        include_dirs = [os.path.join(opencv_dir, 'include')]
        
        # Determine architecture and library path
        lib_path = os.path.join(opencv_dir, 'x64')
        # Find the latest Visual Studio version directory (e.g., vc15, vc16, vc17)
        vs_dirs = [d for d in os.listdir(lib_path) if d.startswith('vc') and os.path.isdir(os.path.join(lib_path, d))]
        if not vs_dirs:
            print(f"ERROR: Could not find Visual Studio lib directory (e.g., 'vc15', 'vc16') in {lib_path}", file=sys.stderr)
            return [], [], []
        
        # Select the latest VS version
        latest_vs_dir = sorted(vs_dirs)[-1]
        library_dirs = [os.path.join(lib_path, latest_vs_dir, 'lib')]

        # Find the opencv_world library file to determine the version number
        lib_files = os.listdir(library_dirs[0])
        world_lib = next((f for f in lib_files if f.startswith('opencv_world') and f.endswith('.lib')), None)

        if not world_lib:
            print(f"ERROR: Could not find 'opencv_world*.lib' in {library_dirs[0]}", file=sys.stderr)
            print("       Ensure you have the compiled OpenCV libraries for C++.", file=sys.stderr)
            return [], [], []
            
        # Extract the library name without the .lib extension
        libraries = [os.path.splitext(world_lib)[0]]

        print(f"Found OpenCV includes: {include_dirs}")
        print(f"Found OpenCV lib dirs: {library_dirs}")
        print(f"Found OpenCV libs: {libraries}")
        return include_dirs, library_dirs, libraries
        
    print("ERROR: Unsupported platform. Could not automatically determine OpenCV paths.", file=sys.stderr)
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
    ext_modules=ext_modules
)