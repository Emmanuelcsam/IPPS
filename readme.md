# Fiber Optic End Face Inspection System

An automated system for detecting and analyzing defects (scratches and digs) in fiber optic end faces using advanced image processing techniques.

## Prerequisites

### Installing UV (Python Environment Manager)
Choose one of the following installation methods:

- **Windows PowerShell:**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS/Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or
  wget -qO- https://astral.sh/uv/install.sh | sh
  ```
- **Via pip:**
  ```bash
  pip install uv
  ```

### One-Line Installation
```bash
uv venv && uv pip install -r requirements.txt pybind11 opencv-python "napari[all]" anomalib torch torchvision scikit-learn numpy joblib scipy omegaconf openvino openvino-dev timm --upgrade && uv run python setup.py build_ext --inplace
```

### Step-by-Step Installation
```bash
uv venv
uv pip install -r requirements.txt
uv pip install pybind11
uv pip install opencv-python
uv pip install "napari[all]"
uv pip install --upgrade anomalib
uv pip install --upgrade timm
uv pip install anomalib torch torchvision scikit-learn numpy opencv-python joblib scipy omegaconf
uv pip install openvino openvino-dev
uv run python setup.py build_ext --inplace
```

## Project Structure

### Test Versions

#### Test13
- **Description:** Fully functional Python implementation with comprehensive defect detection
- **Performance:** Computationally intensive but operationally complete
- **Documentation:** See "Fiber Optic Defect Detector Slideshow.pdf" for detailed overview

#### Test14
- **Description:** C++ optimized version of Test13
- **Performance:** Significantly faster execution while maintaining functionality
- **Usage:** Run with `uv run python run.py`

#### Test15 (NEW)
- **Description:** Reduces photos to matrices and assess defects based on direct computations 
- **Components:**
  - Intensity matrix conversion
  - Difference analysis with heatmap generation
  - Advanced defect detection algorithms
  - Interactive inspection workflow

### Test15 Modules

| Module | Description |
|--------|-------------|
| `main.py` | Interactive inspection system with unified workflow |
| `image_to_matrix.py` | Converts fiber images to pixel intensity matrices |
| `heatmap.py` | Generates difference heatmaps highlighting large variations |
| `inverse_heatmap.py` | Creates heatmaps emphasizing small differences |
| `pixel_defects.py` | Detects scratches (lines) and digs (dots) while excluding fiber rings |
| `matrix_to_img.py` | Converts matrix data back to images |
| `batch_matrix_to_img.py` | Batch conversion of inspection outputs to images |

### Additional Resources

- **High-quality-ellipse-detection:** https://github.com/AlanLuSun/High-quality-ellipse-detection
- **segdec-net-jim2019:** https://github.com/skokec/segdec-net-jim2019

## Directory Structure

```
├── output/              # Contains all scan results
├── sample1/             # Single sample image for individual scan
├── sample2/             # Multiple sample images for batch scans
├── scratchdataset/      # Scratch image samples for training algorithms
├── Test13/              # Python implementation
├── Test14LATEST/        # C++ optimized version
└── Test15/              # New modular pipeline
```

## Usage

### Test14 (Recommended for Production)
```bash
cd Test14LATEST
uv run python run.py
```

Follow the interactive prompts to:
1. Select input directory with fiber images
2. Choose output directory for results
3. Configure fiber specifications
4. Select processing profile (deep_inspection or fast_scan)
5. Choose fiber type for pass/fail rules

### Test15 (New Modular Approach)

#### Interactive Mode
```bash
cd Test15
uv run python main.py
```

The system will guide you through:
- Image selection
- Quick scan with defaults or detailed configuration
- Real-time processing with progress updates
- Comprehensive report generation

#### Batch Mode
```bash
# Create batch configuration
echo '{
  "image_path": "path/to/your/image.png",
  "output_dir": "inspection_output",
  "intensity_method": "luminance",
  "output_formats": ["numpy", "json"],
  "difference_method": "gradient_magnitude",
  "neighborhood": "8-connected",
  "colormap": "black_to_red",
  "highlight_all": true,
  "threshold": 0.0,
  "gamma": 0.5,
  "blur": 0,
  "num_rings": 2,
  "min_scratch_length": 20,
  "min_dig_area": 10,
  "enhancement_factor": 2.0,
  "create_visualizations": true
}' > batch_config.json

# Run batch inspection
uv run python main.py --batch batch_config.json
```

#### Individual Module Usage
```bash
# Convert image to intensity matrix
uv run python image_to_matrix.py input_image.png -o output_dir -f numpy json csv_coords -v

# Generate difference heatmap
uv run python heatmap.py intensity_matrix.npy -o output_dir -c black_to_red --highlight-all -v

# Detect defects
uv run python pixel_defects.py --json analysis.json --image heatmap.png -o output_dir

# Convert results back to images
uv run python matrix_to_img.py intensity_matrix.csv -o reconstructed.png -c hot -v
```

## Configuration

### Test14 Configuration (config.json)
- **processing_profiles:** Algorithm parameters for different inspection modes
- **zone_definitions_iec61300_3_35:** Pass/fail rules for different fiber types
- **reporting:** Output generation settings

### Test15 Configuration Options
- **Intensity Methods:** luminance, average, max, min
- **Difference Methods:** gradient_magnitude, max_neighbor, sobel, laplacian, canny_strength
- **Color Maps:** black_to_red, black_red_yellow, heat, custom, highlight
- **Enhancement:** Adjustable parameters for detecting faint defects

## Output Files

### Test14 Outputs
- `*_annotated.png`: Original image with highlighted defects and zones
- `*_report.csv`: Detailed defect listing with properties
- `*_histogram.png`: Angular distribution of defects

### Test15 Outputs
- `*_intensity_matrix.npy/json/csv`: Pixel intensity data
- `*_heatmap.png`: Difference visualization
- `*_defects_detected.png`: Blue-highlighted scratches and digs
- `*_defect_analysis.json`: Comprehensive defect data
- `*_report.png`: Multi-panel analysis visualization
- `inspection_report.txt`: Summary report

## Key Features

- **Multi-Algorithm Fusion:** Combines multiple detection methods for accuracy
- **Automatic Ring Detection:** Identifies and excludes fiber core/cladding boundaries
- **Small Defect Enhancement:** Special algorithms to detect faint defects
- **Interactive Visualization:** Real-time analysis with napari viewer (Test14)
- **Performance Optimization:** C++ acceleration for critical operations
- **IEC 61300-3-35 Compliance:** Industry-standard pass/fail criteria

## Dependencies

- OpenCV (with C++ development libraries for Test14)
- NumPy, Pandas, Matplotlib
- scikit-image, scikit-learn
- PyTorch (for advanced detection algorithms)
- napari (for interactive visualization)
- pybind11 (for C++ bindings)

## Notes

- Ensure all files and dependencies are in the same directory
- Test14 requires C++ compiler for optimal performance
- Test15 provides more modular control over the inspection process
- Both systems support batch processing of multiple images
