# Image Processing Pipeline GUI

A powerful and flexible GUI for building custom image processing pipelines.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the setup script (already done):
   ```bash
   python setup_gui.py
   ```

3. Start the GUI:
   ```bash
   python image_processor_gui.py
   ```

## Adding Your Scripts

### Method 1: Direct Copy (if scripts are already compatible)
1. Copy your .py files to the `scripts` directory
2. Make sure each script has a `process_image(image)` function
3. The GUI will automatically detect and load them

### Method 2: Using Script Cleaner (for scripts with hardcoded paths)
1. Place your original scripts in a separate directory
2. Run the script cleaner:
   ```bash
   python script_cleaner.py --source your_scripts_dir --output scripts
   ```

### Method 3: Manual Conversion
Create a wrapper for your existing functions:

```python
'''Description of what your script does'''
import cv2
import numpy as np

def process_image(image: np.ndarray, param1: int = 10) -> np.ndarray:
    '''
    Process the image.
    
    Args:
        image: Input image
        param1: Description of parameter
        
    Returns:
        Processed image
    '''
    # Your processing code here
    result = your_existing_function(image, param1)
    return result
```

## Directory Structure

- `scripts/` - Place your image processing functions here
- `scripts/cleaned/` - Automatically cleaned versions of scripts
- `images/` - Store your test images here
- `output/` - Save processed images here
- `pipelines/` - Save/load pipeline configurations here

## Features

- **Dynamic Function Loading**: Automatically detects all scripts in the scripts directory
- **Visual Pipeline Builder**: Drag and drop to reorder processing steps
- **Parameter Editing**: Double-click pipeline items to edit parameters
- **Real-time Feedback**: See which script is currently executing
- **Zoom & Pan**: Mouse wheel to zoom, middle-click to pan
- **Search & Filter**: Find functions by name or category
- **Save/Load Pipelines**: Save your processing workflows for reuse

## Tips

1. Scripts with Unicode errors will be automatically cleaned
2. The GUI shows the exact filename being executed for debugging
3. Use Ctrl+Mouse Wheel for zooming
4. Middle-click and drag to pan around the image
5. Pipeline configurations are saved as JSON files

## Troubleshooting

If a script fails to load:
1. Check the console for error messages
2. Ensure the script has a `process_image` function
3. Try running the script cleaner on it
4. Check that all imports are available

Happy image processing!
