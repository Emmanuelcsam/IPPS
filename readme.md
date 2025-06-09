
**ENSURE ALL PREREQUISTES INSTALLED**

- **Installing UV**
  - powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  - curl -LsSf https://astral.sh/uv/install.sh \| sh
  - wget -qO- https://astral.sh/uv/install.sh \| sh
  - pip install uv
- **One line install**
  -`uv venv && uv pip install -r requirements.txt pybind11 opencv-python "napari[all]" anomalib torch torchvision scikit-learn numpy joblib scipy omegaconf openvino openvino-dev timm --upgrade && uv run python setup.py build_ext --inplace`
- **Steps to install and run program with uv**
  - `uv venv`
  - `uv pip install -r requirements.txt`
  - `uv pip install pybind11`
  - `uv pip install opencv-python`
  - `uv pip install "napari[all]"`
  - `uv pip install --upgrade anomalib`
  - `uv pip install --upgrade timm`
  - `uv pip install anomalib torch torchvision scikit-learn numpy opencv-python joblib scipy omegaconf`
  - `uv pip install openvino openvino-dev`
  - `uv run python setup.py build_ext --inplace`
  - `uv run run.py`

**READ ALL DOCUMENTATION**

**ONLY WORKS WHEN DEPENDENCIES ARE IN THE SAME DIRECTORY AS ALL FILES AND ALL CORRELATING FILES ARE IN THE SAME DIRECTORY**

## **Files**

### **Additional Programs**

High-quality-ellipse-detection https://github.com/AlanLuSun/High-quality-ellipse-detection

segdec-net-jim2019
https://github.com/skokec/segdec-net-jim2019

### **output**
`Contains all scan results`


### **sample1**
`Contains one single sample image at a time, for individual scan`

### **sample2**
`Contains multiple sample images at a time, for batch scans`

### **scratchdateset**
`Contains scratch image samples for training algorthims`

### **Test1.7**
`Contains first seven interation of program`

`Test 3 is most functional but lacks batch scan ability`

`Test 5 is least functional`

### **Test8.12**
`Test 8 is a slighlty alterated iteration of Test 7`

`Test 9-12 are AI iterations of the program`

### **Test13**
`Test 13 is computationally slow but operationally functional`

`Review "Fiber Optic Defect Dector Slideshow.pdf" for full understanding`

### **Test14LATEST**
`Test 14 is Test 13 recreated and converted to C++`

`Computationally fast and latest iteration`


- **Current Program Architecture (Test13 & 14)**  
  - `main.py`  
    - `config_loader.json`: Loads configuration parameters from a JSON file.  
    - `calibration.py`: Determines the physical scale (µm/pixel).  
  - `image_processing.py`  
    - Preprocessing  
    - Localization  
    - Zone Generation  
    - Defect Detection  
  - `analysis.py`  
    - Characterization  
    - Classification  
    - Applying Pass/Fail Rules  
  - `reporting.py`  
    - Annotated Images  
    - CSV Reports  
    - Polar Histograms  








## **Project Structure**

The project is organized into several key modules:

| File | Description |
| :---- | :---- |
| run.py | Interactive Runner: The main entry point to run the inspection via a user-friendly command-line interface. |
| image\_processing.py | Core Engine: Handles image loading, preprocessing, fiber localization, and the multi-algorithm defect detection fusion. |
| analysis.py | Rule Engine: Characterizes defects found by the processing engine and applies pass/fail criteria. |
| reporting.py | Output Generator: Creates all output files, including annotated images, CSV reports, and plots. |
| config.json | Central Configuration: A JSON file to control all parameters, from algorithm settings to pass/fail rules. |
| advanced\_scratch\_detection.py | Contains specialized, advanced algorithms for detecting linear scratch defects. |
| advanced\_visualization.py | Implements the interactive napari viewer for detailed analysis. |
| accelerator.cpp | The C++ source code for performance-critical analysis functions. |
| setup.py | A standard Python script to compile the accelerator.cpp extension using pybind11. |
| requirements.txt | A list of all required Python packages for the project. |

## **Setup and Installation**

### **1\. Prerequisites**

* Python 3.8+  
* A C++ compiler (e.g., GCC on Linux, MSVC on Windows)  
* OpenCV with C++ development libraries installed.

### **2\. Install Python Dependencies**

Install all required Python packages using pip:

pip install \-r requirements.txt

### **3\. Compile the C++ Accelerator (Optional, Recommended)**

For a significant performance boost, compile the C++ extension. This requires pybind11 (installed via requirements.txt) and a properly configured C++ environment with access to OpenCV headers and libraries.

* On Linux: Ensure pkg-config can find opencv4.  
* On Windows: Set the OPENCV\_DIR environment variable to your OpenCV build directory (e.g., C:\\opencv\\build).

Run the following command in the project's root directory:

python setup.py build\_ext \--inplace

If the compilation is successful, a accelerator module will be created, and the program will use it automatically. If it fails or you skip this step, the application will fall back to the slower, pure Python implementation.

## **How to Run**

The inspection process is started through the interactive runner run.py.

python run.py

The script will guide you through the following setup steps:

1. Input Directory: The full path to the folder containing the fiber images you want to inspect.  
2. Output Directory: The full path where reports and logs will be saved. It will be created if it doesn't exist.  
3. Fiber Specifications (Optional): You can provide the known core and cladding diameters in microns to enable more accurate, micron-based zone definitions.  
4. Processing Profile: Choose between deep\_inspection (default) and fast\_scan.  
5. Fiber Type: Specify the key from config.json that corresponds to the pass/fail rules you want to apply (e.g., single\_mode\_pc).

## **Configuration**

The entire inspection process is controlled by config.json. Key sections include:

* processing\_profiles: Defines the algorithms and parameters for different inspection modes (fast\_scan, deep\_inspection).  
* algorithm\_parameters: Global settings for specific image processing algorithms.  
* zone\_definitions\_iec61300\_3\_35: Contains the pass/fail rules for different fiber types. You can define zones (Core, Cladding, etc.) and specify rules such as maximum defect counts and sizes (in microns).  
* reporting: Controls the output generation, such as image DPI and font sizes.

## **Output**

After an inspection run, the specified output directory will contain:

* A sub-directory for each processed image.  
* Inside each sub-directory:  
  * \*\_annotated.png: The original image with defects and zones highlighted.  
  * \*\_report.csv: A detailed CSV file listing every detected defect and its properties.  
  * \*\_histogram.png: A polar plot showing the angular distribution of defects.  
* A main log file for the entire run.

## **Key Dependencies**

* [OpenCV](https://opencv.org/): Core computer vision and image processing tasks.  
* [NumPy](https://numpy.org/): Fundamental package for scientific computing.  
* [Matplotlib](https://matplotlib.org/): Used for generating plots like the polar histogram.  
* [Pandas](https://pandas.pydata.org/): Used for creating and managing the CSV defect reports.  
* [Napari](https://napari.org/): For the optional interactive visualization.  
* [pybind11](https://github.com/pybind/pybind11): For creating the C++/Python bindings.
