# Specify the minimum version of CMake required.
cmake_minimum_required(VERSION 3.10)

# Define the project name.
project(FiberDefectDetector)

# Set the C++ standard to C++17 and enforce it.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Dependencies ---
# Find OpenCV version 4 or newer.
# The 'REQUIRED' keyword will cause CMake to fail if OpenCV is not found.
# We list the specific components our project uses.
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui ximgproc)

# Find Eigen version 3.3 or newer.
# Eigen is a header-only library, so 'NO_MODULE' is used.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


# --- Configure Executable ---
# Add the executable target and specify its source file.
add_executable(fiber_defect_detector fiber_defect_detector.cpp)

# --- Link Dependencies ---
# Link the necessary libraries to our executable.
# Using target_link_libraries with modern targets (like OpenCV::opencv and Eigen3::Eigen)
# is the preferred way, as it also handles include directories automatically.
target_link_libraries(fiber_defect_detector 
    PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# --- Compiler Flags ---
# Add optimization flags for GCC and Clang compilers for better performance.
# -O3 is a high level of optimization.
# -march=native optimizes for the specific CPU architecture of the machine you are compiling on.
# -Wall and -Wextra enable helpful compiler warnings.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fiber_defect_detector PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# --- Installation (Optional) ---
# This rule specifies that the compiled executable should be installed
# into the 'bin' directory when 'make install' is run.
install(TARGETS fiber_defect_detector DESTINATION bin)

# Print a message to the user upon configuration.
message(STATUS "Configuration complete. To build, run 'make' in your build directory.")


