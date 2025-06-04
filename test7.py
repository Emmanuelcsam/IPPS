# main_inspector.py

# Import all necessary libraries for image processing, numerical operations, plotting, and file system management.
import cv2  # OpenCV library for all core computer vision tasks.
import numpy as np  # NumPy library for numerical and array operations.
import matplotlib.pyplot as plt  # Matplotlib for generating all plots, graphs, and histograms.
import os  # Operating system module to interact with the file system (e.g., creating directories).
import csv  # CSV module for writing data to comma-separated value files.
import datetime  # Datetime module to timestamp events and measure processing durations.
from pathlib import Path  # Pathlib for object-oriented filesystem paths, making path manipulation easier and more readable.
import math # Math module for advanced mathematical functions.

class AdvancedFiberInspector:
    """
    Encapsulates the entire logic for fiber optic end face inspection.
    This class handles everything from loading images to detecting defects and generating final reports.
    """

    def __init__(self, core_dia_um=None, clad_dia_um=None):
        """
        Initializes the inspector instance.
        - Sets up user-provided fiber specifications.
        - Determines the operating mode (microns vs. pixels).
        - Initializes state variables.
        """
        # Store the user-provided core diameter in microns. Default is None if not provided.
        self.core_dia_um = core_dia_um
        # Store the user-provided cladding diameter in microns. Default is None if not provided.
        self.clad_dia_um = clad_dia_um
        # Initialize the pixel-to-micron conversion ratio. It will be calculated later if specs are provided.
        self.microns_per_pixel = None
        # Set the initial operating mode. This will be updated based on the provided specifications.
        self.mode = 'PIXEL_ONLY'  # Default mode assumes no physical dimensions are known.
        # Check if the cladding diameter was provided to determine the operating mode.
        if self.clad_dia_um is not None:
            # If cladding diameter is known, we can calculate the micron ratio later.
            self.mode = 'MICRON_CALCULATED'
            # Print the determined operating mode to the console for user feedback.
            print(f"INFO: Operating in '{self.mode}' mode. Micron measurements will be calculated.")
        else:
            # If no physical dimensions are known, the script will work exclusively in pixel units.
            print(f"INFO: Operating in '{self.mode}' mode. All measurements will be in pixels.")

    def _timestamped_print(self, message):
        """
        A utility function to print messages with a timestamp.
        This is crucial for performance tracking and debugging the process flow.
        """
        # Get the current time and format it as a string.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Print the formatted timestamp followed by the message.
        print(f"[{current_time}] {message}")

    def _find_fiber_center_and_zones(self, gray_image):
        """
        Detects the primary concentric circles of the fiber (cladding and core)
        and calculates their properties.
        """
        # Announce the start of the circle detection process.
        self._timestamped_print("Starting automatic circle recognition and zoning...")
        # Apply a Gaussian blur to the image to reduce noise and improve the accuracy of circle detection.
        # A 9x9 kernel provides a moderate amount of smoothing.
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

        # Use the Hough Circle Transform to detect circles in the blurred image.
        # This function is powerful but requires careful parameter tuning.
        circles = cv2.HoughCircles(
            blurred_image,                          # The input image must be grayscale.
            cv2.HOUGH_GRADIENT,                     # Specifies the detection method. HOUGH_GRADIENT is the only one currently implemented.
            dp=1.2,                                 # Inverse ratio of the accumulator resolution to the image resolution.
            minDist=100,                            # Minimum distance between the centers of detected circles.
            param1=60,                              # The higher threshold for the internal Canny edge detector.
            param2=50,                              # The accumulator threshold for circle centers at the detection stage.
            minRadius=int(gray_image.shape[1] / 8), # Minimum circle radius to detect.
            maxRadius=int(gray_image.shape[1] / 3)  # Maximum circle radius to detect.
        )

        # Initialize a dictionary to store the results for the detected zones.
        zone_data = {'cladding': None, 'core': None, 'ferrule': None, 'microns_per_pixel': None}
        # Proceed only if the HoughCircles function returned at least one circle.
        if circles is not None:
            # Convert the circle parameters (x, y, radius) to integers.
            circles = np.uint16(np.around(circles))
            # Assume the largest detected circle is the cladding.
            # We find the index of the circle with the maximum radius.
            cladding_circle = circles[0, np.argmax(circles[0, :, 2])]
            # Extract the center coordinates (cx, cy) and radius (r) of the cladding circle.
            cx, cy, r = int(cladding_circle[0]), int(cladding_circle[1]), int(cladding_circle[2])

            # If a known cladding diameter was provided, calculate the microns-per-pixel ratio.
            if self.clad_dia_um is not None:
                # The formula is the known physical diameter divided by the detected pixel diameter.
                self.microns_per_pixel = self.clad_dia_um / (2 * r)
                # Store the calculated ratio in the zone data dictionary.
                zone_data['microns_per_pixel'] = self.microns_per_pixel
                # Print the calculated ratio for user verification.
                self._timestamped_print(f"Calculated conversion ratio: {self.microns_per_pixel:.4f} µm/pixel.")

            # Store the cladding information in the zone data dictionary.
            zone_data['cladding'] = {'center': (cx, cy), 'radius_px': r}

            # Now, detect the core. The core is the dark circular region *inside* the cladding.
            # We create a mask for the cladding area to search only within it.
            cladding_mask = np.zeros_like(gray_image)
            # Draw a filled circle on the mask corresponding to the detected cladding.
            cv2.circle(cladding_mask, (cx, cy), r, 255, -1)
            # Use the mask to create an image showing only the cladding area.
            cladding_only_image = cv2.bitwise_and(gray_image, gray_image, mask=cladding_mask)

            # To find the core, we threshold the cladding area to isolate the dark core region.
            # Otsu's binarization automatically determines the optimal threshold value.
            _, core_thresh = cv2.threshold(cladding_only_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # After thresholding, remove everything outside the cladding.
            core_thresh = cv2.bitwise_and(core_thresh, core_thresh, mask=cladding_mask)

            # Find contours in the thresholded image to identify the core shape.
            contours, _ = cv2.findContours(core_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Proceed if any contours were found.
            if contours:
                # The largest contour within the cladding should be the core.
                core_contour = max(contours, key=cv2.contourArea)
                # Fit a minimum enclosing circle to the core contour to get its radius and center.
                (core_cx, core_cy), core_r = cv2.minEnclosingCircle(core_contour)
                # Store the core information in the zone data dictionary.
                zone_data['core'] = {'center': (int(core_cx), int(core_cy)), 'radius_px': int(core_r)}

            # Define the ferrule as the region just outside the cladding.
            # We define its radius as a multiple of the cladding radius. This can be adjusted.
            ferrule_radius = int(r * 2.0)
            # Store the ferrule information in the zone data dictionary.
            zone_data['ferrule'] = {'center': (cx, cy), 'radius_px': ferrule_radius}
        # Announce the completion of the zoning process.
        self._timestamped_print("Zone detection complete.")
        # Return the dictionary containing all the geometric data for the fiber zones.
        return zone_data

    def _create_zone_masks(self, image_shape, zone_data):
        """
        Creates binary masks for each identified fiber zone (core, cladding, ferrule).
        """
        # Create a blank array (all zeros) with the same dimensions as the original image.
        core_mask = np.zeros(image_shape, dtype=np.uint8)
        # Create a blank array for the cladding mask.
        cladding_mask = np.zeros(image_shape, dtype=np.uint8)
        # Create a blank array for the ferrule mask.
        ferrule_mask = np.zeros(image_shape, dtype=np.uint8)

        # Check if the cladding data was successfully extracted.
        if zone_data.get('cladding'):
            # Get the center and radius of the cladding.
            cx, cy = zone_data['cladding']['center']
            # Get the radius of the cladding.
            clad_r = zone_data['cladding']['radius_px']
            # Draw a filled white circle on the cladding mask.
            cv2.circle(cladding_mask, (cx, cy), clad_r, 255, -1)

            # Check if the core data was successfully extracted.
            if zone_data.get('core'):
                # Get the radius of the core.
                core_r = zone_data['core']['radius_px']
                # Draw a filled white circle on the core mask.
                cv2.circle(core_mask, (cx, cy), core_r, 255, -1)
                # To get only the cladding region (the donut shape), subtract the core area from the cladding mask.
                cv2.circle(cladding_mask, (cx, cy), core_r, 0, -1)

            # Check if the ferrule data was successfully extracted.
            if zone_data.get('ferrule'):
                # Get the radius of the ferrule.
                ferrule_r = zone_data['ferrule']['radius_px']
                # Draw a filled white circle on the ferrule mask.
                cv2.circle(ferrule_mask, (cx, cy), ferrule_r, 255, -1)
                # Subtract the cladding area to get only the ferrule region.
                cv2.circle(ferrule_mask, (cx, cy), clad_r, 0, -1)
        # Return a dictionary containing the generated masks.
        return {'core': core_mask, 'cladding': cladding_mask, 'ferrule': ferrule_mask}

    def _detect_region_defects_do2mr(self, zone_image, zone_mask):
        """
        Detects region-based defects (dirt, pits, etc.) using a method
        inspired by the "Difference of Min-Max Ranking Filtering" (DO2MR) paper.
        """
        # Create a structuring element for morphological operations. A 5x5 square is a good starting point.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # Apply a minimum filter (erosion) to find the darkest pixel in each neighborhood.
        min_filtered = cv2.erode(zone_image, kernel)
        # Apply a maximum filter (dilation) to find the brightest pixel in each neighborhood.
        max_filtered = cv2.dilate(zone_image, kernel)
        # Calculate the residual image by subtracting the min-filtered from the max-filtered image.
        # This highlights areas of high local contrast, which are characteristic of defect edges.
        residual = cv2.subtract(max_filtered, min_filtered)
        # Apply a median blur to the residual image to reduce salt-and-pepper noise.
        residual_blurred = cv2.medianBlur(residual, 5)
        # Apply Otsu's binarization to the residual image to create a binary mask of potential defects.
        # This method automatically finds an optimal global threshold.
        _, defect_mask = cv2.threshold(residual_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Ensure the detected defects are only within the specified zone by applying the zone mask.
        defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=zone_mask)
        # Use a morphological opening operation to remove small, isolated noise pixels from the defect mask.
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Find the contours of the remaining objects in the defect mask.
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Return the list of detected defect contours.
        return contours

    def _detect_scratches_lei(self, zone_image, zone_mask):
        """
        Detects linear scratches using a method inspired by the "Linear Enhancement
        Inspector" (LEI) paper.
        """
        # Apply Histogram Equalization to the grayscale zone image to maximize contrast, making faint scratches more visible.
        enhanced_image = cv2.equalizeHist(zone_image)
        # Apply the zone mask to the enhanced image to focus only on the relevant area.
        enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=zone_mask)
        # Initialize a blank image to store the maximum response from all filter orientations.
        max_response_map = np.zeros_like(enhanced_image, dtype=np.float32)

        # Iterate through a series of angles to detect scratches at different orientations.
        # A 15-degree increment covers a good range of orientations without being too computationally expensive.
        for angle in range(0, 180, 15):
            # Define the length of the linear kernel.
            kernel_length = 15
            # Create a horizontal line kernel.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            # Get the rotation matrix to orient the kernel to the current angle.
            rot_matrix = cv2.getRotationMatrix2D((kernel_length // 2, 0), angle, 1.0)
            # Apply the rotation to the kernel.
            rotated_kernel = cv2.warpAffine(kernel, rot_matrix, (kernel_length, kernel_length))
            # Normalize the kernel.
            rotated_kernel = rotated_kernel.astype(np.float32) / np.sum(rotated_kernel)
            # Convolve the enhanced image with the rotated linear kernel.
            # This produces a response map where high values indicate features aligned with the kernel.
            response = cv2.filter2D(enhanced_image.astype(np.float32), -1, rotated_kernel)
            # Update the max response map with the pixel-wise maximum values from the current orientation's response.
            max_response_map = np.maximum(max_response_map, response)

        # Normalize the max response map to the 0-255 range for thresholding.
        cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX)
        # Convert the map to an 8-bit integer type.
        response_8u = max_response_map.astype(np.uint8)
        # Apply a high threshold to the response map to segment the scratches.
        # Otsu's method is again used for its adaptive nature.
        _, scratch_mask = cv2.threshold(response_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply morphological closing to connect broken segments of scratches.
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # Find the contours of the detected scratches.
        contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Return the list of scratch contours.
        return contours

    def _analyze_and_classify_defects(self, defects, zone_data):
        """
        Analyzes a list of detected defect contours to calculate their properties.
        """
        # Initialize an empty list to store the detailed information for each defect.
        analyzed_defects = []
        # Get the pixel-to-micron conversion ratio from the zone data.
        ratio = zone_data.get('microns_per_pixel')
        # Iterate through each defect found.
        for i, (contour, defect_type, zone_name) in enumerate(defects):
            # Calculate the area of the defect contour in pixels.
            area_px = cv2.contourArea(contour)
            # Set a minimum area threshold to filter out insignificant noise.
            if area_px < 5:
                # If the area is too small, skip to the next contour.
                continue
            # Calculate the moments of the contour, which are needed to find the centroid.
            M = cv2.moments(contour)
            # Calculate the centroid (center of mass) of the defect.
            # Add a small epsilon to the denominator to avoid division by zero.
            cx = int(M['m10'] / (M['m00'] + 1e-5))
            # Calculate the y-coordinate of the centroid.
            cy = int(M['m01'] / (M['m00'] + 1e-5))
            # Get the bounding box coordinates for the defect.
            x, y, w, h = cv2.boundingRect(contour)

            # Initialize the area in microns to "N/A".
            area_um2 = "N/A"
            # If the conversion ratio is available, calculate the area in square microns.
            if ratio:
                # The formula is pixel area multiplied by the square of the ratio.
                area_um2 = area_px * (ratio ** 2)

            # Append a dictionary with all the defect's properties to the list.
            analyzed_defects.append({
                'Defect_ID': f"{zone_name[:3].upper()}-{i+1}", # Create a unique ID for the defect.
                'Zone': zone_name, # The zone where the defect was found.
                'Type': defect_type, # The type of defect (Region or Scratch).
                'Centroid_X_px': cx, # X-coordinate of the centroid in pixels.
                'Centroid_Y_px': cy, # Y-coordinate of the centroid in pixels.
                'Area_px2': area_px, # Area in square pixels.
                'Area_um2': area_um2, # Area in square microns (if available).
                'Bounding_Box': (x, y, w, h), # The bounding box of the defect.
                'Confidence_Score': 1.0 # Placeholder for a future, more complex confidence model.
            })
        # Return the list of dictionaries, each containing detailed info about a defect.
        return analyzed_defects

    def _generate_visual_report(self, image, classified_defects, zone_data, output_dir, filename_base):
        """
        Generates and saves visual reports: an annotated image and a polar histogram of defects.
        """
        # Create a copy of the original image to draw annotations on, preserving the original.
        annotated_image = image.copy()
        # --- Draw Zone Circles ---
        # Check if cladding data is available.
        if zone_data.get('cladding'):
            # Get cladding center and radius.
            center = zone_data['cladding']['center']
            # Get cladding radius.
            clad_r = zone_data['cladding']['radius_px']
            # Draw the cladding circle on the image in green.
            cv2.circle(annotated_image, center, clad_r, (0, 255, 0), 2)
            # Check if core data is available.
            if zone_data.get('core'):
                # Get core radius.
                core_r = zone_data['core']['radius_px']
                # Draw the core circle on the image in blue.
                cv2.circle(annotated_image, center, core_r, (255, 0, 0), 2)

        # --- Draw Defect Bounding Boxes and Labels ---
        # Initialize lists to store defect coordinates for the polar plot.
        radii, thetas, colors = [], [], []
        # Get the center of the fiber, which is the origin for the polar plot.
        fiber_center = zone_data.get('cladding', {}).get('center')
        # Iterate through each classified defect.
        for defect in classified_defects:
            # Get the bounding box coordinates.
            x, y, w, h = defect['Bounding_Box']
            # Set the color for the bounding box based on the defect type.
            color = (0, 0, 255) if defect['Type'] == 'Scratch' else (0, 255, 255) # Red for scratch, yellow for region.
            # Draw the rectangle on the annotated image.
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            # Create a label for the defect.
            label = f"{defect['Defect_ID']}"
            # Put the label text on the image above the bounding box.
            cv2.putText(annotated_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Prepare Data for Polar Histogram ---
            # Proceed only if the fiber center was found.
            if fiber_center:
                # Get the defect's centroid.
                cx, cy = defect['Centroid_X_px'], defect['Centroid_Y_px']
                # Calculate the defect's position relative to the fiber center.
                dx = cx - fiber_center[0]
                # Calculate the y-offset.
                dy = cy - fiber_center[1]
                # Calculate the radial distance (r) from the center.
                radii.append(math.sqrt(dx**2 + dy**2))
                # Calculate the angle (theta) using arctan2 for correct quadrant placement.
                thetas.append(math.atan2(dy, dx))
                # Assign a color for the plot point based on defect type.
                colors.append('red' if defect['Type'] == 'Scratch' else 'yellow')

        # --- Save the Annotated Image ---
        # Construct the full path for the output file.
        annotated_image_path = output_dir / f"{filename_base}_annotated.jpg"
        # Save the image with all the annotations.
        cv2.imwrite(str(annotated_image_path), annotated_image)

        # --- Generate and Save the Polar Histogram ---
        # Proceed only if there are defects to plot.
        if radii:
            # Create a new figure and a polar subplot.
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            # Create a scatter plot of the defects using their polar coordinates.
            ax.scatter(thetas, radii, c=colors, alpha=0.75)
            # Set the title of the plot.
            ax.set_title('Defect Distribution', va='bottom')
            # Invert the r-axis so that the core (r=0) is at the center.
            ax.set_rlim(max(radii) if radii else 1, 0)
            # Construct the full path for the output histogram file.
            histogram_path = output_dir / f"{filename_base}_histogram.png"
            # Save the figure to a file.
            plt.savefig(histogram_path)
            # Close the figure to free up memory.
            plt.close(fig)

    def _generate_csv_report(self, classified_defects, output_dir, filename_base):
        """
        Generates and saves a detailed CSV report for a single image.
        """
        # Construct the full path for the output CSV file.
        report_path = output_dir / f"{filename_base}_report.csv"
        # Check if any defects were found to create a report for.
        if not classified_defects:
            # If no defects, we can either create an empty file or just skip. Here we skip.
            return

        # Open the specified file in write mode.
        with open(report_path, 'w', newline='') as csvfile:
            # Define the headers for the CSV file based on the keys of the defect dictionary.
            fieldnames = classified_defects[0].keys()
            # Create a DictWriter object to write dictionaries to the CSV file.
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write the header row to the CSV file.
            writer.writeheader()
            # Write all the defect data rows to the file.
            writer.writerows(classified_defects)

    def inspect_image(self, image_path, output_dir):
        """
        Orchestrates the entire inspection pipeline for a single image.
        """
        # Record the start time for this specific image to measure processing duration.
        start_time = datetime.datetime.now()
        # Create a Path object for easier file name manipulation.
        p = Path(image_path)
        # Extract the filename without its extension to use as a base for output files.
        filename_base = p.stem
        # Print a message indicating which image is currently being processed.
        self._timestamped_print(f"--- Processing image: {p.name} ---")

        # Load the image from the specified path.
        image = cv2.imread(str(image_path))
        # Check if the image was loaded successfully.
        if image is None:
            # If the image failed to load, print an error and return a failure status.
            self._timestamped_print(f"ERROR: Failed to load image at {image_path}")
            # Return a dictionary indicating the error.
            return {'Image_Filename': p.name, 'Total_Defects': 'N/A', 'Processing_Time_s': 'N/A', 'Status': 'Error'}

        # Convert the loaded BGR image to grayscale for analysis.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the fiber zones (core, cladding, ferrule).
        zone_data = self._find_fiber_center_and_zones(gray_image)
        # If the primary zone (cladding) was not found, the image cannot be processed further.
        if not zone_data or not zone_data.get('cladding'):
            # Print an error message.
            self._timestamped_print(f"ERROR: Could not identify fiber structure in {p.name}.")
            # Return a failure status.
            return {'Image_Filename': p.name, 'Total_Defects': 'N/A', 'Processing_Time_s': 'N/A', 'Status': 'Zoning Failed'}

        # Create binary masks for each zone.
        masks = self._create_zone_masks(gray_image.shape, zone_data)
        # Announce the start of the defect detection phase.
        self._timestamped_print("Starting defect detection...")
        # Initialize an empty list to hold all detected defects before classification.
        all_detected_defects = []
        # Loop through the zones we want to analyze (core and cladding).
        for zone_name in ['core', 'cladding']:
            # Skip if the mask for the current zone was not created.
            if np.sum(masks[zone_name]) == 0:
                # Continue to the next zone.
                continue
            # Apply the region-based defect detection method (DO2MR).
            region_defects = self._detect_region_defects_do2mr(gray_image, masks[zone_name])
            # Extend the master list with the found contours, tagging them as 'Region'.
            all_detected_defects.extend([(c, 'Region', zone_name) for c in region_defects])
            # Apply the scratch detection method (LEI).
            scratch_defects = self._detect_scratches_lei(gray_image, masks[zone_name])
            # Extend the master list with the found contours, tagging them as 'Scratch'.
            all_detected_defects.extend([(c, 'Scratch', zone_name) for c in scratch_defects])
        # Announce the completion of the defect detection phase.
        self._timestamped_print("Defect detection complete.")

        # Analyze and classify the raw defect contours.
        classified_defects = self._analyze_and_classify_defects(all_detected_defects, zone_data)
        # Announce the start of the report generation phase.
        self._timestamped_print("Generating reports...")
        # Generate the visual reports (annotated image and polar plot).
        self._generate_visual_report(image, classified_defects, zone_data, output_dir, filename_base)
        # Generate the detailed CSV report for this image.
        self._generate_csv_report(classified_defects, output_dir, filename_base)
        # Announce the completion of the report generation phase.
        self._timestamped_print(f"Reports for {p.name} saved to '{output_dir}'.")
        # Record the end time for this image's processing.
        end_time = datetime.datetime.now()
        # Calculate the total processing time.
        processing_time = (end_time - start_time).total_seconds()

        # Count the number of defects found in each zone for the summary report.
        core_defects_count = sum(1 for d in classified_defects if d['Zone'] == 'core')
        # Count cladding defects.
        cladding_defects_count = sum(1 for d in classified_defects if d['Zone'] == 'cladding')
        # Return a summary dictionary for this image.
        return {
            'Image_Filename': p.name,
            'Total_Defects': len(classified_defects),
            'Core_Defects': core_defects_count,
            'Cladding_Defects': cladding_defects_count,
            'Processing_Time_s': f"{processing_time:.2f}"
        }

    def process_batch(self, image_dir):
        """
        Processes all images in a given directory.
        """
        # Record the start time for the entire batch operation.
        batch_start_time = datetime.datetime.now()
        # Create a unique output directory name using the current timestamp.
        output_dir = Path(f"inspection_results_{batch_start_time.strftime('%Y%m%d_%H%M%S')}")
        # Create the output directory.
        output_dir.mkdir(exist_ok=True)
        # Announce the start of the batch processing.
        self._timestamped_print(f"Starting batch processing. Results will be saved in '{output_dir}'.")

        # Define the valid image file extensions to look for.
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        # Get a list of all files in the directory that have a valid image extension.
        image_paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in image_extensions]

        # Check if any images were found.
        if not image_paths:
            # If no images were found, print a warning and exit the function.
            self._timestamped_print(f"WARNING: No images found in directory: {image_dir}")
            # Return without processing.
            return

        # Initialize a list to store the summary results from each image.
        summary_results = []
        # Loop through each image path found in the directory.
        for image_path in image_paths:
            # Call the single-image inspection pipeline.
            result = self.inspect_image(image_path, output_dir)
            # If the processing was successful, append the result to the summary list.
            if result:
                # Add the result to the list.
                summary_results.append(result)

        # --- Compile and Save the Final Summary Report ---
        # Check if any images were successfully processed.
        if summary_results:
            # Construct the path for the final summary CSV file.
            summary_report_path = output_dir / "summary_report.csv"
            # Open the summary report file in write mode.
            with open(summary_report_path, 'w', newline='') as csvfile:
                # Define the headers for the summary report.
                fieldnames = summary_results[0].keys()
                # Create a DictWriter object.
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # Write the header row.
                writer.writeheader()
                # Write all the summary data.
                writer.writerows(summary_results)
            # Announce the completion of the summary report.
            self._timestamped_print(f"Batch summary report saved to '{summary_report_path}'.")
        # Calculate the total time taken for the entire batch.
        total_time = (datetime.datetime.now() - batch_start_time).total_seconds()
        # Announce the completion of the entire batch process.
        self._timestamped_print(f"--- Batch processing complete in {total_time:.2f} seconds. ---")

def main():
    """
    The main function to drive the script. Handles user input and orchestrates the inspection process.
    """
    # Print a welcome message for the user.
    print("=" * 60)
    # Print the title of the application.
    print(" Advanced Automated Optical Fiber End Face Inspector")
    # Print another line for visual separation.
    print("=" * 60)

    # --- Get Directory Path from User ---
    # Prompt the user to enter the path to the directory containing the images.
    image_dir = input("Enter the path to the directory with fiber images: ").strip()
    # Check if the provided path is actually a directory.
    if not Path(image_dir).is_dir():
        # If not, print an error message and exit the script.
        print(f"ERROR: The path '{image_dir}' is not a valid directory.")
        # Exit the program.
        return

    # --- Get Fiber Specifications from User (Optional) ---
    # Initialize diameter variables to None.
    core_diameter_um, cladding_diameter_um = None, None
    # Ask the user if they want to provide known physical specifications.
    provide_specs = input("Do you want to provide known fiber specifications (in microns)? (y/n): ").strip().lower()
    # If the user answers 'yes'.
    if provide_specs == 'y':
        # Start a loop to get valid input for the core diameter.
        while True:
            # Try to get and convert the core diameter to a float.
            try:
                # Prompt for the core diameter.
                core_diameter_um = float(input("Enter core diameter in microns (e.g., 9, 50, 62.5): "))
                # If successful, break the loop.
                break
            # Catch a ValueError if the input is not a valid number.
            except ValueError:
                # Print an error message.
                print("Invalid input. Please enter a number.")
        # Start a loop to get valid input for the cladding diameter.
        while True:
            # Try to get and convert the cladding diameter to a float.
            try:
                # Prompt for the cladding diameter.
                cladding_diameter_um = float(input("Enter cladding diameter in microns (e.g., 125): "))
                # If successful, break the loop.
                break
            # Catch a ValueError if the input is not a valid number.
            except ValueError:
                # Print an error message.
                print("Invalid input. Please enter a number.")
    else:
        # If the user chooses not to provide specs, inform them about the default behavior.
        print("Proceeding without physical specifications. All measurements will be in pixels.")

    # --- Instantiate and Run the Inspector ---
    # Create an instance of the main inspector class, passing the user-provided specs.
    inspector = AdvancedFiberInspector(core_dia_um=core_diameter_um, clad_dia_um=cladding_diameter_um)
    # Start the batch processing by calling the `process_batch` method with the image directory.
    inspector.process_batch(image_dir)

# This standard Python construct ensures that the main() function is called only when the script is executed directly.
if __name__ == "__main__":
    # Call the main function.
    main()