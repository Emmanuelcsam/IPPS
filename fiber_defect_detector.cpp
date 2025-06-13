// --- Standard and Third-Party Libraries ---
// Eigen headers must be included BEFORE the OpenCV eigen header.
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <complex>
#include <iostream>
#include <memory>
#include <map>

// Use float for image processing for better compatibility and performance.
using real_t = float;

class FiberOpticDefectDetector {
private:
    bool debug;
    cv::Mat original;
    int height, width;
    std::map<std::string, cv::Mat> intermediateResults;

    // Constants
    static constexpr real_t PI = 3.14159265358979323846f;
    static constexpr real_t EPSILON = 1e-7f;

public:
    struct DefectResults {
        cv::Mat scratches;
        cv::Mat digs;
        cv::Mat combined;
        std::map<std::string, cv::Mat> intermediate;
    };

    FiberOpticDefectDetector(bool debug = false) : debug(debug) {}

    DefectResults detectDefects(const cv::Mat& image) {
        cv::Mat grayImage;
        if (image.channels() == 3) {
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        } else {
            grayImage = image.clone();
        }

        // Store original
        original = grayImage.clone();
        height = grayImage.rows;
        width = grayImage.cols;

        // Step 1: Advanced preprocessing
        cv::Mat preprocessed = advancedPreprocessing(grayImage);

        // Step 2: Multi-method scratch detection
        cv::Mat scratchMask = detectScratchesMultimethod(preprocessed);

        // Step 3: Multi-method dig detection
        cv::Mat digMask = detectDigsMultimethod(preprocessed);

        // Step 4: Refinement using variational methods
        cv::Mat refinedScratches = refineWithVariationalMethods(scratchMask, preprocessed);
        cv::Mat refinedDigs = refineWithVariationalMethods(digMask, preprocessed);

        // Step 5: False positive reduction
        cv::Mat finalScratches = reduceFalsePositivesScratches(refinedScratches);
        cv::Mat finalDigs = reduceFalsePositivesDigs(refinedDigs, preprocessed);

        // Combine results
        cv::Mat combined;
        cv::bitwise_or(finalScratches, finalDigs, combined);

        DefectResults results;
        results.scratches = finalScratches;
        results.digs = finalDigs;
        results.combined = combined;
        if (debug) {
            results.intermediate = intermediateResults;
        }

        return results;
    }

private:
    cv::Mat advancedPreprocessing(const cv::Mat& image) {
        // Convert to CV_32F (float) instead of CV_64F for broader function support.
        cv::Mat imageFloat;
        image.convertTo(imageFloat, CV_32F, 1.0f/255.0f);

        // 1. Anisotropic diffusion (Perona-Malik)
        cv::Mat diffused = anisotropicDiffusion(imageFloat);

        // 2. Total Variation denoising (simplified version using bilateral filter)
        // This function now receives a supported format (CV_32F).
        cv::Mat tvDenoised;
        cv::bilateralFilter(diffused, tvDenoised, -1, 0.1, 5.0);

        // 3. Coherence-enhancing diffusion
        cv::Mat coherenceEnhanced = coherenceEnhancingDiffusion(tvDenoised);

        // 4. Adaptive histogram equalization
        cv::Mat enhanced;
        cv::normalize(coherenceEnhanced, coherenceEnhanced, 0, 255, cv::NORM_MINMAX);
        coherenceEnhanced.convertTo(enhanced, CV_8U);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(enhanced, enhanced);

        if (debug) {
            intermediateResults["preprocessed"] = enhanced;
        }

        enhanced.convertTo(enhanced, CV_32F, 1.0f/255.0f);
        return enhanced;
    }

    cv::Mat anisotropicDiffusion(const cv::Mat& image, int iterations = 10,
                                 real_t kappa = 30.0f, real_t gamma = 0.15f) {
        cv::Mat img = image.clone();

        for (int iter = 0; iter < iterations; ++iter) {
            cv::Mat nablaN, nablaS, nablaE, nablaW;
            cv::Mat grad_kernel_N = (cv::Mat_<real_t>(3,3) << 0, -1, 0, 0, 1, 0, 0, 0, 0);
            cv::Mat grad_kernel_S = (cv::Mat_<real_t>(3,3) << 0, 0, 0, 0, 1, 0, 0, -1, 0);
            cv::Mat grad_kernel_E = (cv::Mat_<real_t>(3,3) << 0, 0, 0, 0, 1, -1, 0, 0, 0);
            cv::Mat grad_kernel_W = (cv::Mat_<real_t>(3,3) << 0, 0, 0, -1, 1, 0, 0, 0, 0);

            cv::filter2D(img, nablaN, CV_32F, grad_kernel_N);
            cv::filter2D(img, nablaS, CV_32F, grad_kernel_S);
            cv::filter2D(img, nablaE, CV_32F, grad_kernel_E);
            cv::filter2D(img, nablaW, CV_32F, grad_kernel_W);

            cv::Mat cN, cS, cE, cW;
            cv::exp(-(nablaN.mul(nablaN)) / (kappa*kappa), cN);
            cv::exp(-(nablaS.mul(nablaS)) / (kappa*kappa), cS);
            cv::exp(-(nablaE.mul(nablaE)) / (kappa*kappa), cE);
            cv::exp(-(nablaW.mul(nablaW)) / (kappa*kappa), cW);

            img += gamma * (cN.mul(nablaN) + cS.mul(nablaS) + cE.mul(nablaE) + cW.mul(nablaW));
        }
        return img;
    }

    cv::Mat coherenceEnhancingDiffusion(const cv::Mat& image, int iterations = 5) {
        cv::Mat img = image.clone();

        for (int iter = 0; iter < iterations; ++iter) {
            std::vector<cv::Mat> J = computeStructureTensor(img);
            auto [eigenvals, eigenvecs] = eigenDecomposition2x2(J);
            std::vector<cv::Mat> D = computeDiffusionTensor(eigenvals, eigenvecs);
            img = applyTensorDiffusion(img, D);
        }
        return img;
    }

    std::vector<cv::Mat> computeStructureTensor(const cv::Mat& image, real_t sigma = 1.0f) {
        cv::Mat Ix, Iy;
        cv::Sobel(image, Ix, CV_32F, 1, 0);
        cv::Sobel(image, Iy, CV_32F, 0, 1);

        cv::Mat Jxx = Ix.mul(Ix);
        cv::Mat Jxy = Ix.mul(Iy);
        cv::Mat Jyy = Iy.mul(Iy);

        cv::GaussianBlur(Jxx, Jxx, cv::Size(0, 0), sigma);
        cv::GaussianBlur(Jxy, Jxy, cv::Size(0, 0), sigma);
        cv::GaussianBlur(Jyy, Jyy, cv::Size(0, 0), sigma);

        return {Jxx, Jxy, Jyy};
    }

    std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> eigenDecomposition2x2(
        const std::vector<cv::Mat>& J) {
        cv::Mat a = J[0], b = J[1], c = J[2];
        cv::Mat trace = a + c;
        cv::Mat det = a.mul(c) - b.mul(b);
        cv::Mat discriminant;
        cv::sqrt(cv::max(trace.mul(trace) - 4*det, 0), discriminant);
        cv::Mat lambda1 = 0.5f * (trace + discriminant);
        cv::Mat lambda2 = 0.5f * (trace - discriminant);
        cv::Mat v1x = lambda1 - c;
        cv::Mat v1y = b;
        cv::Mat norm1;
        cv::sqrt(v1x.mul(v1x) + v1y.mul(v1y) + EPSILON, norm1);
        v1x = v1x / norm1;
        v1y = v1y / norm1;
        cv::Mat v2x = -v1y;
        cv::Mat v2y = v1x;
        return {{lambda1, lambda2}, {v1x, v1y, v2x, v2y}};
    }

    std::vector<cv::Mat> computeDiffusionTensor(const std::vector<cv::Mat>& eigenvals,
                                                const std::vector<cv::Mat>& eigenvecs,
                                                real_t alpha = 0.001f) {
        cv::Mat lambda1 = eigenvals[0], lambda2 = eigenvals[1];
        cv::Mat coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + EPSILON));
        coherence = coherence.mul(coherence);
        cv::Mat c1 = cv::Mat::ones(lambda1.size(), lambda1.type()) * alpha;
        cv::Mat c2;
        cv::exp(-1.0f / (coherence + EPSILON), c2);
        c2 = alpha + (1.0f - alpha) * c2;
        cv::Mat D11 = c1.mul(eigenvecs[0].mul(eigenvecs[0])) + c2.mul(eigenvecs[2].mul(eigenvecs[2]));
        cv::Mat D12 = c1.mul(eigenvecs[0].mul(eigenvecs[1])) + c2.mul(eigenvecs[2].mul(eigenvecs[3]));
        cv::Mat D22 = c1.mul(eigenvecs[1].mul(eigenvecs[1])) + c2.mul(eigenvecs[3].mul(eigenvecs[3]));
        return {D11, D12, D22};
    }

    cv::Mat applyTensorDiffusion(const cv::Mat& image, const std::vector<cv::Mat>& D, real_t dt = 0.1f) {
        cv::Mat Ixx, Iyy, Ixy;
        cv::Sobel(image, Ixx, CV_32F, 2, 0);
        cv::Sobel(image, Iyy, CV_32F, 0, 2);
        cv::Sobel(image, Ixy, CV_32F, 1, 1);
        cv::Mat div = D[0].mul(Ixx) + 2.0f * D[1].mul(Ixy) + D[2].mul(Iyy);
        return image + dt * div;
    }

    cv::Mat detectScratchesMultimethod(const cv::Mat& image) {
        std::vector<cv::Mat> methods;
        methods.push_back(hessianRidgeDetection(image));
        methods.push_back(frangiVesselness(image));
        methods.push_back(computePhaseCongruency(image));
        methods.push_back(radonLineDetection(image));
        methods.push_back(directionalFilterBank(image));
        std::vector<real_t> weights = {0.25f, 0.25f, 0.2f, 0.15f, 0.15f};
        cv::Mat combined = cv::Mat::zeros(image.size(), CV_32F);
        for (size_t i = 0; i < methods.size(); ++i) combined += weights[i] * methods[i];
        cv::Scalar mean, stddev;
        cv::meanStdDev(combined, mean, stddev);
        double threshold = mean[0] + 2.0 * stddev[0];
        cv::Mat scratchMask;
        cv::threshold(combined, scratchMask, threshold, 255, cv::THRESH_BINARY);
        scratchMask.convertTo(scratchMask, CV_8U);
        scratchMask = morphologicalScratchRefinement(scratchMask);
        if (debug) intermediateResults["scratch_combined"] = combined;
        return scratchMask;
    }

    cv::Mat hessianRidgeDetection(const cv::Mat& image, const std::vector<real_t>& scales = {1.0f, 2.0f, 3.0f}) {
        cv::Mat ridgeResponse = cv::Mat::zeros(image.size(), CV_32F);
        for (real_t scale : scales) {
            cv::Mat smoothed;
            cv::GaussianBlur(image, smoothed, cv::Size(0, 0), scale);
            cv::Mat Hxx, Hyy, Hxy;
            cv::Sobel(smoothed, Hxx, CV_32F, 2, 0);
            cv::Sobel(smoothed, Hyy, CV_32F, 0, 2);
            cv::Sobel(smoothed, Hxy, CV_32F, 1, 1);
            cv::Mat trace = Hxx + Hyy;
            cv::Mat det = Hxx.mul(Hyy) - Hxy.mul(Hxy);
            cv::Mat discriminant;
            cv::sqrt(cv::max(trace.mul(trace) - 4.0f*det, 0), discriminant);
            cv::Mat lambda1 = 0.5f * (trace + discriminant);
            cv::Mat lambda2 = 0.5f * (trace - discriminant);
            cv::Mat Rb;
            cv::divide(cv::abs(lambda1), (cv::abs(lambda2) + EPSILON), Rb);
            cv::Mat S;
            cv::sqrt(lambda1.mul(lambda1) + lambda2.mul(lambda2), S);
            real_t beta = 0.5f;
            real_t c = 0.5f * cv::norm(S, cv::NORM_INF);
            if (c < EPSILON) c = EPSILON;
            cv::Mat response;
            cv::exp(-Rb.mul(Rb) / (2.0f*beta*beta), response);
            cv::Mat term2;
            cv::exp(-S.mul(S) / (2.0f*c*c), term2);
            response = response.mul(cv::Scalar::all(1) - term2);
            response.setTo(0, lambda2 >= 0);
            cv::max(ridgeResponse, scale*scale * response, ridgeResponse);
        }
        return ridgeResponse;
    }

    cv::Mat frangiVesselness(const cv::Mat& image) {
        cv::Mat vesselness = cv::Mat::zeros(image.size(), CV_32F);
        std::vector<real_t> scales = {1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
        for (real_t scale : scales) {
            cv::Mat smoothed, Hxx, Hyy, Hxy;
            cv::GaussianBlur(image, smoothed, cv::Size(0, 0), scale);
            cv::Sobel(smoothed, Hxx, CV_32F, 2, 0);
            cv::Sobel(smoothed, Hyy, CV_32F, 0, 2);
            cv::Sobel(smoothed, Hxy, CV_32F, 1, 1);
            cv::Mat tmp;
            cv::sqrt((Hxx - Hyy).mul(Hxx - Hyy) + 4.0f*Hxy.mul(Hxy), tmp);
            cv::Mat l1 = 0.5f * (Hxx + Hyy + tmp);
            cv::Mat l2 = 0.5f * (Hxx + Hyy - tmp);
            cv::Mat mask = cv::abs(l1) > cv::abs(l2);
            cv::Mat temp;
            l1.copyTo(temp); l2.copyTo(l1, mask); temp.copyTo(l2, mask);
            cv::Mat Rb;
            cv::divide(cv::abs(l1), (cv::abs(l2) + EPSILON), Rb);
            cv::Mat S;
            cv::sqrt(l1.mul(l1) + l2.mul(l2), S);
            real_t beta = 0.5f, gamma = 15.0f;
            cv::Mat v;
            cv::exp(-Rb.mul(Rb) / (2.0f*beta*beta), v);
            cv::Mat term2;
            cv::exp(-S.mul(S) / (2.0f*gamma*gamma), term2);
            v = v.mul(cv::Scalar::all(1) - term2);
            v.setTo(0, l2 >= 0);
            cv::max(vesselness, v, vesselness);
        }
        return vesselness;
    }

    cv::Mat computePhaseCongruency(const cv::Mat& image, int nscale = 4, int norient = 6) {
        cv::Mat PC = cv::Mat::zeros(image.size(), CV_32F);
        real_t wavelength = 6.0f;
        for (int s = 0; s < nscale; ++s) {
            real_t lambda = wavelength * std::pow(2, s);
            for (int o = 0; o < norient; ++o) {
                real_t angle = o * PI / norient;
                cv::Mat kernel = cv::getGaborKernel(cv::Size(21, 21), 4.0f, angle, lambda, 0.5f, 0, CV_32F);
                cv::Mat response;
                cv::filter2D(image, response, CV_32F, kernel);
                PC += cv::abs(response);
            }
        }
        cv::normalize(PC, PC, 0, 1, cv::NORM_MINMAX);
        return PC;
    }

    cv::Mat radonLineDetection(const cv::Mat& image) {
        cv::Mat imageU8;
        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
        image.convertTo(imageU8, CV_8U);
        cv::Mat edges;
        cv::Canny(imageU8, edges, 50, 150, 3);
        std::vector<cv::Vec2f> lines;
        cv::HoughLines(edges, lines, 1, CV_PI/180, 80);
        cv::Mat lineMask = cv::Mat::zeros(image.size(), CV_32F);
        if (!lines.empty()) {
            for (const auto& line : lines) drawLineFromRadon(lineMask, line[0], line[1]);
        }
        return lineMask;
    }

    void drawLineFromRadon(cv::Mat& mask, real_t rho, real_t theta) {
        cv::Point pt1, pt2;
        real_t a = cos(theta), b = sin(theta);
        real_t x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cv::line(mask, pt1, pt2, cv::Scalar(1.0f), 1, cv::LINE_AA);
    }

    cv::Mat directionalFilterBank(const cv::Mat& image, int n_orientations = 16) {
        cv::Mat responseMap = cv::Mat::zeros(image.size(), CV_32F);
        for (int i = 0; i < n_orientations; ++i) {
            real_t angle = i * PI / n_orientations;
            cv::Mat kernel = createSteerableFilter(angle);
            cv::Mat response;
            cv::filter2D(image, response, CV_32F, kernel);
            cv::Mat suppressed = directionalNMS(response, angle + PI/2.0f);
            cv::max(responseMap, suppressed, responseMap);
        }
        return responseMap;
    }

    cv::Mat createSteerableFilter(real_t angle, int size = 15, real_t sigma = 2.0f) {
        cv::Mat kernel(size, size, CV_32F);
        int center = size / 2;
        real_t sum = 0.0f;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                real_t dx = x - center, dy = y - center;
                real_t x_rot = dx * std::cos(angle) + dy * std::sin(angle);
                real_t y_rot = -dx * std::sin(angle) + dy * std::cos(angle);
                real_t g = std::exp(-(x_rot*x_rot + y_rot*y_rot) / (2.0f*sigma*sigma));
                real_t val = (x_rot*x_rot/(sigma*sigma) - 1.0f) * g / (2.0f*PI*pow(sigma,4));
                kernel.at<real_t>(y, x) = val;
                sum += val;
            }
        }
        kernel -= (sum / (size*size));
        return kernel;
    }

    cv::Mat directionalNMS(const cv::Mat& response, real_t angle) {
        cv::Mat suppressed = response.clone();
        real_t dx = std::cos(angle), dy = std::sin(angle);
        for (int y = 1; y < response.rows - 1; ++y) {
            for (int x = 1; x < response.cols - 1; ++x) {
                real_t val = response.at<real_t>(y, x);
                real_t val1 = bilinearInterpolate(response, x + dx, y + dy);
                real_t val2 = bilinearInterpolate(response, x - dx, y - dy);
                if (val < val1 || val < val2) suppressed.at<real_t>(y, x) = 0;
            }
        }
        return suppressed;
    }

    real_t bilinearInterpolate(const cv::Mat& img, real_t x, real_t y) {
        int x0 = static_cast<int>(std::floor(x)), x1 = x0 + 1;
        int y0 = static_cast<int>(std::floor(y)), y1 = y0 + 1;
        if (x0 < 0 || x1 >= img.cols || y0 < 0 || y1 >= img.rows) return 0.0f;
        real_t wa = (x1 - x) * (y1 - y), wb = (x - x0) * (y1 - y);
        real_t wc = (x1 - x) * (y - y0), wd = (x - x0) * (y - y0);
        return wa * img.at<real_t>(y0, x0) + wb * img.at<real_t>(y0, x1) + wc * img.at<real_t>(y1, x0) + wd * img.at<real_t>(y1, x1);
    }

    cv::Mat morphologicalScratchRefinement(const cv::Mat& mask) {
        cv::Mat refined = cv::Mat::zeros(mask.size(), CV_8U);
        for (int angle = 0; angle < 180; angle += 15) {
            cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 1));
            cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(7, 0), angle, 1);
            cv::warpAffine(se, se, rotationMatrix, se.size());
            cv::Mat closed, opened;
            cv::morphologyEx(mask, closed, cv::MORPH_CLOSE, se);
            cv::morphologyEx(closed, opened, cv::MORPH_OPEN, se);
            cv::bitwise_or(refined, opened, refined);
        }
        return refined;
    }

    cv::Mat detectDigsMultimethod(const cv::Mat& image) {
        std::vector<cv::Mat> methods;
        methods.push_back(scaleNormalizedLoG(image));
        methods.push_back(determinantOfHessian(image));
        methods.push_back(mserDetection(image));
        methods.push_back(morphologicalBlobDetection(image));
        methods.push_back(lbpBlobDetection(image));
        std::vector<real_t> weights = {0.3f, 0.25f, 0.2f, 0.15f, 0.1f};
        cv::Mat combined = cv::Mat::zeros(image.size(), CV_32F);
        for (size_t i = 0; i < methods.size(); ++i) combined += weights[i] * methods[i];
        cv::Scalar mean, stddev;
        cv::meanStdDev(combined, mean, stddev);
        double threshold = mean[0] + 1.5 * stddev[0];
        cv::Mat digMask;
        cv::threshold(combined, digMask, threshold, 255, cv::THRESH_BINARY);
        digMask.convertTo(digMask, CV_8U);
        digMask = shapeBasedFiltering(digMask);
        if (debug) intermediateResults["dig_combined"] = combined;
        return digMask;
    }

    cv::Mat scaleNormalizedLoG(const cv::Mat& image) {
        cv::Mat blobResponse = cv::Mat::zeros(image.size(), CV_32F);
        std::vector<real_t> scales = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        for (real_t scale : scales) {
            cv::Mat smoothed, log;
            cv::GaussianBlur(image, smoothed, cv::Size(), scale, scale);
            cv::Laplacian(smoothed, log, CV_32F);
            log = -log * scale * scale;
            cv::max(blobResponse, log, blobResponse);
        }
        return blobResponse;
    }

    cv::Mat determinantOfHessian(const cv::Mat& image) {
        return hessianRidgeDetection(image);
    }

    cv::Mat mserDetection(const cv::Mat& image) {
        cv::Mat imgU8;
        // Input `image` is CV_32F in range [0,1]. Convert directly to CV_8U.
        image.convertTo(imgU8, CV_8U, 255.0);
        cv::Ptr<cv::MSER> mser = cv::MSER::create(5, 60, 14400, 0.25, 0.2);
        std::vector<std::vector<cv::Point>> regions;
        // FIX: Create a named variable for the bounding boxes to pass as an lvalue reference.
        std::vector<cv::Rect> bboxes;
        mser->detectRegions(imgU8, regions, bboxes);
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_32F);
        for (const auto& region : regions) {
            cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{region}, cv::Scalar(1.0f));
        }
        return mask;
    }

    cv::Mat morphologicalBlobDetection(const cv::Mat& image) {
        cv::Mat inverted;
        cv::subtract(cv::Scalar::all(1.0f), image, inverted);
        cv::Mat marker = inverted - 0.1f, mask = inverted, reconstruction;
        cv::morphologyEx(marker, reconstruction, cv::MORPH_DILATE, cv::Mat());
        cv::min(reconstruction, mask, reconstruction);
        cv::Mat regionalMax = inverted - reconstruction;
        cv::Mat blobMask;
        cv::threshold(regionalMax, blobMask, 0.01f, 1.0f, cv::THRESH_BINARY);
        blobMask.convertTo(blobMask, CV_8U);
        cv::Mat labeled;
        int numLabels = cv::connectedComponents(blobMask, labeled, 8);
        for (int label = 1; label < numLabels; ++label) {
            cv::Mat labelMask = (labeled == label);
            int area = cv::countNonZero(labelMask);
            if (area < 5 || area > 1000) {
                 blobMask.setTo(0, labelMask);
                 continue;
            }
            cv::Moments m = cv::moments(labelMask, true);
            if (m.m00 > 0) {
                double mu20 = m.mu20 / m.m00, mu02 = m.mu02 / m.m00, mu11 = m.mu11 / m.m00;
                double lambda_denom = std::sqrt(pow(mu20 - mu02, 2) + 4 * mu11 * mu11);
                double lambda1 = 0.5 * (mu20 + mu02 + lambda_denom);
                double lambda2 = 0.5 * (mu20 + mu02 - lambda_denom);
                if (lambda1 > EPSILON) {
                    if (std::sqrt(1.0 - lambda2 / lambda1) > 0.8) blobMask.setTo(0, labelMask);
                }
            }
        }
        blobMask.convertTo(blobMask, CV_32F);
        return blobMask;
    }

    cv::Mat lbpBlobDetection(const cv::Mat& image) {
        cv::Mat anomalyScore;
        cv::Mat imgU8;
        image.convertTo(imgU8, CV_8U, 255.0f);
        cv::boxFilter(imgU8, anomalyScore, CV_32F, cv::Size(15,15));
        cv::normalize(anomalyScore, anomalyScore, 0, 1, cv::NORM_MINMAX);
        return anomalyScore;
    }

    cv::Mat shapeBasedFiltering(const cv::Mat& mask) {
        cv::Mat filtered = mask.clone();
        cv::Mat labeled;
        int numLabels = cv::connectedComponents(mask, labeled);
        for (int label = 1; label < numLabels; ++label) {
            cv::Mat labelMask = (labeled == label);
            double area = cv::countNonZero(labelMask);
            if (area > 5 && area < 2000) {
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(labelMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                if (!contours.empty()) {
                    double perimeter = cv::arcLength(contours[0], true);
                    double circularity = (4 * PI * area) / (perimeter * perimeter + EPSILON);
                    std::vector<cv::Point> hull;
                    cv::convexHull(contours[0], hull);
                    double solidity = area / (cv::contourArea(hull) + EPSILON);
                    if (circularity < 0.3 || solidity < 0.4) filtered.setTo(0, labelMask);
                }
            } else {
                 filtered.setTo(0, labelMask);
            }
        }
        return filtered;
    }

    cv::Mat refineWithVariationalMethods(const cv::Mat& mask, const cv::Mat& image) {
        if(cv::countNonZero(mask) == 0) return mask;
        cv::Mat phi;
        mask.convertTo(phi, CV_32F, 1.0f/255.0f);
        phi = (phi * 2.0f) - 1.0f;
        cv::Mat refined = chanVeseEvolution(phi, image, 30);
        cv::Mat result;
        cv::threshold(refined, result, 0, 255, cv::THRESH_BINARY);
        result.convertTo(result, CV_8U);
        return result;
    }

    cv::Mat chanVeseEvolution(cv::Mat phi, const cv::Mat& image, int iterations,
                              real_t mu = 0.2f, real_t lambda1 = 1.0f, real_t lambda2 = 1.0f) {
        phi = phi.clone();
        for (int iter = 0; iter < iterations; ++iter) {
            cv::Mat H_phi = (phi >= 0);
            H_phi.convertTo(H_phi, CV_8U, 255);
            double c1 = cv::mean(image, H_phi)[0];
            double c2 = cv::mean(image, ~H_phi)[0];
            cv::Mat term1 = -lambda1 * (image - c1).mul(image - c1);
            cv::Mat term2 = lambda2 * (image - c2).mul(image - c2);
            cv::Mat phi_x, phi_y;
            cv::Sobel(phi, phi_x, CV_32F, 1, 0, 1);
            cv::Sobel(phi, phi_y, CV_32F, 0, 1, 1);
            cv::Mat norm;
            cv::sqrt(phi_x.mul(phi_x) + phi_y.mul(phi_y), norm);
            cv::Mat nx, ny;
            cv::divide(phi_x, norm + EPSILON, nx);
            cv::divide(phi_y, norm + EPSILON, ny);
            cv::Mat nxx, nyy;
            cv::Sobel(nx, nxx, CV_32F, 1, 0, 1);
            cv::Sobel(ny, nyy, CV_32F, 0, 1, 1);
            cv::Mat curvature = nxx + nyy;
            cv::Mat F = term1 + term2 + mu * curvature;
            phi += 0.1f * F.mul(norm);
        }
        return phi;
    }

    cv::Mat reduceFalsePositivesScratches(const cv::Mat& mask) {
        if (cv::countNonZero(mask) == 0) return mask;
        cv::Mat refined = mask.clone();
        cv::Mat labeled;
        int numLabels = cv::connectedComponents(refined, labeled);
        for (int label = 1; label < numLabels; ++label) {
            cv::Mat labelMask = (labeled == label);
            cv::Mat skeleton;
            cv::ximgproc::thinning(labelMask, skeleton, cv::ximgproc::THINNING_GUOHALL);
            if (cv::countNonZero(skeleton) < 20) {
                refined.setTo(0, labelMask);
                continue;
            }
            std::vector<cv::Point> points;
            cv::findNonZero(labelMask, points);
            if (points.size() < 10) continue;
            cv::PCA pca(cv::Mat(points).reshape(1), cv::Mat(), cv::PCA::DATA_AS_ROW);
            if (pca.eigenvalues.at<real_t>(0) > EPSILON) {
                 if (1.0f - (pca.eigenvalues.at<real_t>(1) / pca.eigenvalues.at<real_t>(0)) < 0.9f) {
                    refined.setTo(0, labelMask);
                 }
            }
        }
        return refined;
    }

    cv::Mat reduceFalsePositivesDigs(const cv::Mat& mask, const cv::Mat& image) {
        if (cv::countNonZero(mask) == 0) return mask;
        cv::Mat refined = mask.clone();
        cv::Mat labeled;
        int numLabels = cv::connectedComponents(refined, labeled);
        for (int label = 1; label < numLabels; ++label) {
            cv::Mat labelMask = (labeled == label);
            if (cv::countNonZero(labelMask) < 5) {
                refined.setTo(0, labelMask);
                continue;
            }
            cv::Mat dilated, boundary;
            cv::dilate(labelMask, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
            boundary = dilated - labelMask;
            if (cv::countNonZero(boundary) > 0 && cv::countNonZero(labelMask) > 0) {
                if (std::abs(cv::mean(image, labelMask)[0] - cv::mean(image, boundary)[0]) < 0.05) {
                    refined.setTo(0, labelMask);
                }
            }
        }
        return refined;
    }
};

void visualizeResults(const cv::Mat& image, const FiberOpticDefectDetector::DefectResults& results) {
    cv::Mat display, overlay, finalDisplay;
    cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image, overlay, cv::COLOR_GRAY2BGR);
    overlay.setTo(cv::Scalar(0, 0, 255), results.scratches);
    overlay.setTo(cv::Scalar(0, 255, 0), results.digs);
    cv::addWeighted(display, 0.7, overlay, 0.3, 0, finalDisplay);
    cv::imshow("Original Image", image);
    cv::imshow("Scratches (Red)", results.scratches);
    cv::imshow("Digs (Green)", results.digs);
    cv::imshow("Combined Mask", results.combined);
    cv::imshow("Final Overlay", finalDisplay);
    int scratchPixels = cv::countNonZero(results.scratches);
    int digPixels = cv::countNonZero(results.digs);
    std::cout << "\n--- Defect Detection Results ---\n";
    std::cout << "Scratch-like defects found: " << scratchPixels << " pixels" << std::endl;
    std::cout << "Dig-like defects found: " << digPixels << " pixels" << std::endl;
    std::cout << "--------------------------------\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return -1;
    }
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << argv[1] << std::endl;
        return -1;
    }
    std::cout << "Image loaded successfully. Starting defect detection..." << std::endl;
    FiberOpticDefectDetector detector(false);
    FiberOpticDefectDetector::DefectResults results = detector.detectDefects(image);
    std::cout << "Defect detection complete. Visualizing results..." << std::endl;
    visualizeResults(image, results);
    return 0;
}
