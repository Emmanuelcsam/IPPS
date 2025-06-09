#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // Required for automatic list/dict conversions
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>

namespace py = pybind11;

/**
 * @brief Performs DO2MR (Difference of Min-Max Ranking) filtering for defect detection.
 * This is a high-performance C++ implementation of the DO2MR algorithm.
 * @param image_buffer A py::array_t<unsigned char> representing the input grayscale image (2D, uint8).
 * @param kernel_size The size of the square structuring element for morphological operations.
 * @param gamma The standard deviation multiplier for sigma-based thresholding.
 * @return A py::array_t<unsigned char> representing the binary defect mask (0 or 255).
 */
py::array_t<unsigned char> do2mr_detection_cpp(py::array_t<unsigned char> image_buffer, int kernel_size, double gamma) {
    py::buffer_info buf = image_buffer.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input image must be a 2D NumPy array.");
    }
    cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::Mat dilated_image, eroded_image;
    cv::dilate(image, dilated_image, kernel);
    cv::erode(image, eroded_image, kernel);
    cv::Mat residual;
    cv::subtract(dilated_image, eroded_image, residual);
    cv::Mat residual_filtered;
    cv::medianBlur(residual, residual_filtered, 3);
    cv::Mat mean, stddev;
    cv::meanStdDev(residual_filtered, mean, stddev, image > 0);
    double mu = mean.at<double>(0);
    double sigma = stddev.at<double>(0);
    double threshold_value = mu + gamma * sigma;
    cv::Mat defect_binary;
    cv::threshold(residual_filtered, defect_binary, threshold_value, 255, cv::THRESH_BINARY);
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(defect_binary, defect_binary, cv::MORPH_OPEN, kernel_open);
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(defect_binary, labels, stats, centroids, 8, CV_32S);
    cv::Mat final_mask = cv::Mat::zeros(image.size(), CV_8U);
    int min_defect_area_px = 5;
    for (int i = 1; i < num_labels; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_defect_area_px) {
            final_mask.setTo(255, labels == i);
        }
    }
    
    // --- FIX 1 START: Correctly create a copy of the cv::Mat data for Python ---
    // The original line had a '.clone()' call on the py::array_t object, which is incorrect.
    // The correct way is to create a py::array_t that owns its own data and copy the cv::Mat content into it.
    py::array_t<unsigned char> result({(py::ssize_t)final_mask.rows, (py::ssize_t)final_mask.cols});
    py::buffer_info result_buf = result.request();
    std::memcpy(result_buf.ptr, final_mask.data, final_mask.total() * final_mask.elemSize());
    return result;
    // --- FIX 1 END ---
}

/**
 * @brief Performs multi-scale DO2MR detection
 * @param image_buffer Input image
 * @param scales Vector of scale factors
 * @param base_kernel_size Base kernel size
 * @param gamma Threshold parameter
 * @return Combined defect mask
 */
py::array_t<unsigned char> multiscale_do2mr_cpp(
    py::array_t<unsigned char> image_buffer,
    std::vector<float> scales,
    int base_kernel_size,
    double gamma
) {
    py::buffer_info buf = image_buffer.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D");
    }
    
    cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    cv::Mat combined_result = cv::Mat::zeros(image.size(), CV_32F);
    
    for (float scale : scales) {
        cv::Mat scaled_image;
        
        if (scale != 1.0) {
            cv::Size new_size(image.cols * scale, image.rows * scale);
            cv::resize(image, scaled_image, new_size, 0, 0, cv::INTER_LINEAR);
        } else {
            scaled_image = image.clone();
        }
        
        // Apply DO2MR at this scale
        int kernel_size = std::max(3, static_cast<int>(base_kernel_size * scale));
        if (kernel_size % 2 == 0) kernel_size++;
        
        // Call existing DO2MR logic
        py::array_t<unsigned char> scale_result_py = do2mr_detection_cpp(
            py::array_t<unsigned char>(
                {scaled_image.rows, scaled_image.cols},
                scaled_image.data
            ),
            kernel_size,
            gamma
        );
        
        // Convert back to cv::Mat
        py::buffer_info scale_buf = scale_result_py.request();
        cv::Mat scale_result(scale_buf.shape[0], scale_buf.shape[1],
                             CV_8UC1, (unsigned char*)scale_buf.ptr);
        
        // Resize back and accumulate
        cv::Mat scale_result_resized;
        if (scale != 1.0) {
            cv::resize(scale_result, scale_result_resized, image.size(),
                       0, 0, cv::INTER_NEAREST);
        } else {
            scale_result_resized = scale_result;
        }
        
        // Weight by scale
        float weight = (scale > 1.0) ? 1.0 / scale : scale;
        cv::Mat weighted;
        scale_result_resized.convertTo(weighted, CV_32F, weight);
        combined_result += weighted;
    }
    
    // Normalize and threshold
    cv::Mat normalized;
    cv::normalize(combined_result, normalized, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat final_result;
    cv::threshold(normalized, final_result, 127, 255, cv::THRESH_BINARY);
    
    // Convert to uint8 and return
    cv::Mat final_uint8;
    final_result.convertTo(final_uint8, CV_8U);
    
    py::array_t<unsigned char> result({final_uint8.rows, final_uint8.cols});
    py::buffer_info result_buf = result.request();
    std::memcpy(result_buf.ptr, final_uint8.data,
                final_uint8.total() * final_uint8.elemSize());
    
    return result;
}

/**
 * @brief Characterizes and classifies defects from a binary mask.
 * This function finds connected components, calculates their properties (area, centroid, aspect ratio, etc.),
 * and determines their location within named zones.
 *
 * @param final_defect_mask A 2D NumPy array (uint8) representing the combined defect mask.
 * @param zone_masks A dictionary mapping zone names (string) to their corresponding 2D NumPy masks (uint8).
 * @param um_per_px The conversion factor from pixels to micrometers. Use 0.0 if not available.
 * @param image_filename The base name of the image file for generating unique defect IDs.
 * @param min_defect_area_px The minimum area in pixels for a component to be considered a defect.
 * @param scratch_aspect_ratio_threshold The aspect ratio above which a defect is classified as a "Scratch".
 *
 * @return A list of dictionaries, where each dictionary represents a characterized defect.
 */
py::list characterize_and_classify_defects_cpp(
    py::array_t<unsigned char> final_defect_mask_buffer,
    py::dict zone_masks_dict,
    double um_per_px,
    std::string image_filename,
    int min_defect_area_px,
    double scratch_aspect_ratio_threshold
) {
    py::buffer_info buf = final_defect_mask_buffer.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input defect mask must be a 2D NumPy array.");
    }
    cv::Mat final_defect_mask(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);

    // Find connected components (defects)
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(final_defect_mask, labels, stats, centroids, 8, CV_32S);

    py::list characterized_defects;
    int defect_id_counter = 0;

    // Process each component (label 0 is the background)
    for (int i = 1; i < num_labels; ++i) {
        int area_px = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area_px < min_defect_area_px) {
            continue;
        }

        defect_id_counter++;
        std::string defect_id_str = image_filename + "_D" + std::to_string(defect_id_counter);

        double centroid_x_px = centroids.at<double>(i, 0);
        double centroid_y_px = centroids.at<double>(i, 1);

        // Create a mask for the individual defect component
        cv::Mat component_mask = (labels == i);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(component_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            continue;
        }
        auto defect_contour = contours[0];

        // Get precise dimensions from the rotated bounding box
        cv::RotatedRect rotated_rect = cv::minAreaRect(defect_contour);
        float width_px = rotated_rect.size.width;
        float height_px = rotated_rect.size.height;
        float aspect_ratio = std::max(width_px, height_px) / (std::min(width_px, height_px) + 1e-6);

        // Classify the defect
        std::string classification = (aspect_ratio >= scratch_aspect_ratio_threshold) ? "Scratch" : "Pit/Dig";

        // Determine the zone
        std::string zone_name = "Unknown";
        for (auto item : zone_masks_dict) {
            std::string current_zone_name = py::str(item.first);
            py::array_t<unsigned char> zone_mask_buffer = py::cast<py::array_t<unsigned char>>(item.second);
            py::buffer_info zone_buf = zone_mask_buffer.request();
            cv::Mat zone_mask_mat(zone_buf.shape[0], zone_buf.shape[1], CV_8U, (unsigned char*)zone_buf.ptr);

            int cx_int = static_cast<int>(centroid_x_px);
            int cy_int = static_cast<int>(centroid_y_px);
            if (cy_int >= 0 && cy_int < zone_mask_mat.rows && cx_int >= 0 && cx_int < zone_mask_mat.cols) {
                if (zone_mask_mat.at<unsigned char>(cy_int, cx_int) > 0) {
                    zone_name = current_zone_name;
                    break;
                }
            }
        }
        
        // Build the defect dictionary to return to Python
        py::dict defect_dict;
        defect_dict["defect_id"] = defect_id_str;
        defect_dict["area_px"] = area_px;
        defect_dict["centroid_x_px"] = centroid_x_px;
        defect_dict["centroid_y_px"] = centroid_y_px;
        defect_dict["classification"] = classification;
        defect_dict["zone"] = zone_name;
        defect_dict["aspect_ratio"] = aspect_ratio;

        // --- FIX 2 START: Explicitly define shape vector for py::array_t constructor ---
        // The original code used an initializer list `{...}` which the compiler could not resolve.
        // Creating an explicit std::vector for the shape is more robust.
        std::vector<py::ssize_t> shape = {(py::ssize_t)defect_contour.size(), 2};
        py::array_t<int> contour_points_py(shape);
        // --- FIX 2 END ---
        
        auto r = contour_points_py.mutable_unchecked<2>();
        for(long j = 0; j < r.shape(0); ++j) {
            r(j, 0) = defect_contour[j].x;
            r(j, 1) = defect_contour[j].y;
        }
        defect_dict["contour_points_px"] = contour_points_py;
        
        // Add dimensions in pixels
        defect_dict["width_px"] = std::min(width_px, height_px);
        defect_dict["height_px"] = std::max(width_px, height_px);

        // Add dimensions in microns if scale is available
        if (um_per_px > 0) {
            defect_dict["length_um"] = std::max(width_px, height_px) * um_per_px;
            defect_dict["width_um"] = std::min(width_px, height_px) * um_per_px;
        }

        characterized_defects.append(defect_dict);
    }
    return characterized_defects;
}


// ------------------ pybind11 Module Definition ------------------
// This block defines the Python module and exposes the C++ functions.
PYBIND11_MODULE(dscope_accelerator, m) {
    m.doc() = "D-Scope Blink C++ Accelerator: High-performance image processing and analysis functions.";

    m.def(
        "do2mr_detection",
        &do2mr_detection_cpp,
        "Performs high-speed DO2MR defect detection.",
        py::arg("image"),
        py::arg("kernel_size") = 5,
        py::arg("gamma") = 1.5
    );

    // ADD THE NEW MODULE DEFINITION HERE
    m.def(
        "multiscale_do2mr",
        &multiscale_do2mr_cpp,
        "Performs multi-scale DO2MR defect detection",
        py::arg("image"),
        py::arg("scales"),
        py::arg("base_kernel_size") = 5,
        py::arg("gamma") = 1.5
    );

    m.def(
        "characterize_and_classify_defects",
        &characterize_and_classify_defects_cpp,
        "Performs high-speed defect characterization and classification from a binary mask.",
        py::arg("final_defect_mask"),
        py::arg("zone_masks"),
        py::arg("um_per_px"),
        py::arg("image_filename"),
        py::arg("min_defect_area_px"),
        py::arg("scratch_aspect_ratio_threshold")
    );
}