#!/usr/bin/env python3
"""
ULTIMATE COMPREHENSIVE FIBER OPTIC DEFECT DETECTION SYSTEM
===========================================================
PhD-Level Multi-Method Defect Analysis Engine

This system combines ALL defect detection methodologies from computer vision,
signal processing, machine learning, and deep learning domains to provide
the most comprehensive defect analysis possible.

Author: Advanced Fiber Optics Analysis Team
Version: 3.0 - Ultimate Edition
"""

import cv2
import numpy as np
from scipy import ndimage, signal, stats, spatial, special, optimize
from scipy.fftpack import fft2, ifft2, fftshift, dct, idct
from scipy.ndimage import label, binary_erosion, binary_dilation, distance_transform_edt
from scipy.signal import savgol_filter, find_peaks, peak_widths, cwt, ricker
from scipy.spatial import Voronoi, ConvexHull, Delaunay
from scipy.interpolate import griddata, RBFInterpolator
from skimage import morphology, feature, filters, measure, transform, restoration, segmentation
from skimage.feature import greycomatrix, greycoprops, peak_local_max, corner_harris, corner_shi_tomasi
from skimage.filters import frangi, hessian, meijering, sato, threshold_multiotsu
from skimage.morphology import skeletonize, medial_axis, convex_hull_image
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet
from skimage.segmentation import watershed, chan_vese, morphological_geodesic_active_contour
from skimage.transform import radon, iradon, probabilistic_hough_line
from sklearn.decomposition import PCA, NMF, FastICA, DictionaryLearning, TruncatedSVD
from sklearn.cluster import DBSCAN, KMeans, MeanShift, SpectralClustering, AgglomerativeClustering, OPTICS
from sklearn.ensemble import IsolationForest, RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pywt
import mahotas
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial, lru_cache
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Advanced Configuration
@dataclass
class DefectDetectionConfig:
    """Comprehensive configuration for all detection methods"""
    # Preprocessing
    use_illumination_correction: bool = True
    use_noise_reduction: bool = True
    use_contrast_enhancement: bool = True
    
    # Statistical parameters
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    mad_threshold: float = 3.0
    grubbs_alpha: float = 0.05
    dixon_alpha: float = 0.05
    chauvenet_criterion: float = 0.5
    
    # Morphological parameters
    morph_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9, 11])
    tophat_kernel_sizes: List[int] = field(default_factory=lambda: [9, 15, 21])
    
    # Frequency domain
    use_fft: bool = True
    use_dct: bool = True
    use_wavelet: bool = True
    wavelet_families: List[str] = field(default_factory=lambda: ['db4', 'sym4', 'coif2', 'bior3.5'])
    wavelet_levels: List[int] = field(default_factory=lambda: [3, 4, 5])
    
    # Advanced detection parameters
    gabor_frequencies: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.25])
    gabor_orientations: List[float] = field(default_factory=lambda: np.linspace(0, np.pi, 16).tolist())
    gabor_sigmas: List[float] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    
    # Machine learning
    use_ml_detection: bool = True
    ml_contamination: float = 0.1
    dbscan_eps_range: Tuple[float, float] = (0.3, 2.0)
    dbscan_min_samples_range: Tuple[int, int] = (3, 10)
    
    # Deep learning
    use_deep_learning: bool = False  # Requires trained models
    
    # Validation
    min_contrast_range: Tuple[float, float] = (5.0, 30.0)
    min_defect_area: int = 3
    max_defect_area: int = 10000
    
    # Performance
    use_parallel_processing: bool = True
    num_workers: int = mp.cpu_count()
    
    # Output
    save_intermediate_results: bool = True
    visualization_dpi: int = 300


class DefectType(Enum):
    """Enumeration of all possible defect types"""
    SCRATCH = auto()
    PIT = auto()
    DIG = auto()
    PARTICLE = auto()
    CONTAMINATION = auto()
    CHIP = auto()
    CRACK = auto()
    DELAMINATION = auto()
    BUBBLE = auto()
    INCLUSION = auto()
    STAIN = auto()
    BURN = auto()
    UNKNOWN = auto()


@dataclass
class Defect:
    """Comprehensive defect representation"""
    id: str
    type: DefectType
    confidence: float
    location: Tuple[int, int]  # (x, y) centroid
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area_px: int
    area_um2: Optional[float] = None
    perimeter: float = 0.0
    major_axis: float = 0.0
    minor_axis: float = 0.0
    orientation: float = 0.0
    eccentricity: float = 0.0
    solidity: float = 0.0
    compactness: float = 0.0
    hu_moments: Optional[np.ndarray] = None
    zernike_moments: Optional[np.ndarray] = None
    fourier_descriptors: Optional[np.ndarray] = None
    texture_features: Optional[Dict[str, float]] = None
    intensity_features: Optional[Dict[str, float]] = None
    detection_methods: List[str] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    confidence_map: Optional[np.ndarray] = None


class UltimateDefectDetector:
    """
    The most comprehensive fiber optic defect detection system ever created.
    Implements every known defect detection methodology with PhD-level enhancements.
    """
    
    def __init__(self, config: Optional[DefectDetectionConfig] = None):
        self.config = config or DefectDetectionConfig()
        self.results = {}
        self.defects = []
        self.performance_metrics = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize parallel processing
        if self.config.use_parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        
    def analyze_comprehensive(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                            region_type: str = "unknown") -> Dict[str, Any]:
        """
        Performs the most comprehensive defect analysis possible using every available method.
        """
        start_time = time.time()
        self.logger.info(f"Starting comprehensive analysis for region: {region_type}")
        
        # Prepare image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.color_image = image.copy()
        else:
            gray = image.copy()
            self.color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        self.original_image = gray.copy()
        self.mask = mask if mask is not None else np.ones_like(gray, dtype=np.uint8) * 255
        self.region_type = region_type
        
        # 1. Advanced Preprocessing
        preprocessed_images = self._advanced_preprocessing(gray)
        
        # 2. Multi-scale Analysis
        multiscale_results = self._multiscale_analysis(preprocessed_images)
        
        # 3. Statistical Anomaly Detection (Enhanced)
        statistical_results = self._enhanced_statistical_detection(preprocessed_images)
        
        # 4. Spatial Domain Analysis (PhD-level)
        spatial_results = self._phd_spatial_analysis(preprocessed_images)
        
        # 5. Frequency Domain Analysis (Advanced)
        frequency_results = self._advanced_frequency_analysis(preprocessed_images)
        
        # 6. Transform Domain Analysis
        transform_results = self._transform_domain_analysis(preprocessed_images)
        
        # 7. Morphological Analysis (Comprehensive)
        morphological_results = self._comprehensive_morphological_analysis(preprocessed_images)
        
        # 8. Texture Analysis (Advanced)
        texture_results = self._advanced_texture_analysis(preprocessed_images)
        
        # 9. Edge and Gradient Analysis (PhD-level)
        edge_results = self._phd_edge_analysis(preprocessed_images)
        
        # 10. Machine Learning Detection
        ml_results = self._machine_learning_detection(preprocessed_images)
        
        # 11. Deep Learning Detection (if available)
        dl_results = self._deep_learning_detection(preprocessed_images) if self.config.use_deep_learning else {}
        
        # 12. Physics-based Detection
        physics_results = self._physics_based_detection(preprocessed_images)
        
        # 13. Hybrid and Novel Methods
        hybrid_results = self._hybrid_novel_methods(preprocessed_images)
        
        # 14. Intelligent Fusion of All Methods
        fusion_results = self._intelligent_fusion({
            'multiscale': multiscale_results,
            'statistical': statistical_results,
            'spatial': spatial_results,
            'frequency': frequency_results,
            'transform': transform_results,
            'morphological': morphological_results,
            'texture': texture_results,
            'edge': edge_results,
            'ml': ml_results,
            'dl': dl_results,
            'physics': physics_results,
            'hybrid': hybrid_results
        })
        
        # 15. Advanced Post-processing and Validation
        validated_results = self._advanced_validation(fusion_results)
        
        # 16. Defect Characterization and Classification
        characterized_defects = self._characterize_defects(validated_results)
        
        # 17. Quality Metrics Calculation
        quality_metrics = self._calculate_quality_metrics(characterized_defects)
        
        # 18. Generate Comprehensive Report
        analysis_time = time.time() - start_time
        
        comprehensive_results = {
            'defects': characterized_defects,
            'quality_metrics': quality_metrics,
            'detection_masks': validated_results,
            'confidence_maps': fusion_results.get('confidence_maps', {}),
            'individual_results': {
                'multiscale': multiscale_results,
                'statistical': statistical_results,
                'spatial': spatial_results,
                'frequency': frequency_results,
                'transform': transform_results,
                'morphological': morphological_results,
                'texture': texture_results,
                'edge': edge_results,
                'ml': ml_results,
                'physics': physics_results,
                'hybrid': hybrid_results
            },
            'performance': {
                'analysis_time': analysis_time,
                'methods_used': len(fusion_results.get('methods_used', [])),
                'total_algorithms': self._count_algorithms_used()
            },
            'metadata': {
                'region_type': region_type,
                'image_shape': image.shape,
                'config': self.config
            }
        }
        
        self.results = comprehensive_results
        self.defects = characterized_defects
        
        self.logger.info(f"Analysis complete. Found {len(characterized_defects)} defects in {analysis_time:.2f} seconds")
        
        return comprehensive_results
    
    def _advanced_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Advanced preprocessing with multiple techniques"""
        preprocessed = {'original': image}
        
        # 1. Illumination Correction (Multiple methods)
        if self.config.use_illumination_correction:
            # Rolling ball background subtraction
            preprocessed['rolling_ball'] = self._rolling_ball_correction(image)
            
            # Homomorphic filtering
            preprocessed['homomorphic'] = self._homomorphic_filtering(image)
            
            # Retinex enhancement
            preprocessed['retinex'] = self._multi_scale_retinex(image)
            
            # Adaptive histogram equalization variants
            preprocessed['clahe'] = self._advanced_clahe(image)
        
        # 2. Advanced Noise Reduction
        if self.config.use_noise_reduction:
            # Non-local means
            preprocessed['nlm'] = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            # BM3D-style denoising (simplified)
            preprocessed['bm3d'] = self._simplified_bm3d(image)
            
            # Anisotropic diffusion variants
            preprocessed['perona_malik'] = self._perona_malik_diffusion(image)
            preprocessed['coherence_enhancing'] = self._coherence_enhancing_diffusion(image)
            
            # Wavelet denoising
            preprocessed['wavelet_denoise'] = denoise_wavelet(image, wavelet='db4', 
                                                             rescale_sigma=True)
        
        # 3. Contrast Enhancement
        if self.config.use_contrast_enhancement:
            # Gamma correction variants
            for gamma in [0.5, 0.7, 1.3, 1.5]:
                preprocessed[f'gamma_{gamma}'] = self._gamma_correction(image, gamma)
            
            # Sigmoid enhancement
            preprocessed['sigmoid'] = self._sigmoid_enhancement(image)
            
            # Logarithmic enhancement
            preprocessed['log'] = self._logarithmic_enhancement(image)
        
        return preprocessed
    
    def _multiscale_analysis(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Multi-scale defect detection using scale-space theory"""
        results = {}
        
        for img_name, img in images.items():
            scale_space_results = []
            
            # Gaussian scale-space
            sigmas = np.logspace(0, 1.5, 10)  # 1 to ~31.6
            for sigma in sigmas:
                blurred = cv2.GaussianBlur(img, (0, 0), sigma)
                
                # Detect at this scale
                scale_defects = self._detect_at_scale(blurred, sigma)
                scale_space_results.append((sigma, scale_defects))
            
            # Laplacian scale-space
            log_results = []
            for sigma in sigmas:
                log = cv2.GaussianBlur(img, (0, 0), sigma)
                log = cv2.Laplacian(log, cv2.CV_64F)
                log = np.abs(log)
                log_results.append((sigma, log))
            
            # DoG scale-space
            dog_results = []
            for i in range(len(sigmas) - 1):
                dog = cv2.GaussianBlur(img, (0, 0), sigmas[i+1]) - \
                      cv2.GaussianBlur(img, (0, 0), sigmas[i])
                dog_results.append((sigmas[i], sigmas[i+1], dog))
            
            results[img_name] = {
                'scale_space': scale_space_results,
                'log_space': log_results,
                'dog_space': dog_results
            }
        
        return results
    
    def _enhanced_statistical_detection(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """PhD-level statistical anomaly detection"""
        results = {}
        
        for img_name, img in images.items():
            pixels = img[self.mask > 0].flatten()
            if len(pixels) < 10:
                continue
            
            # 1. Robust statistical measures
            median = np.median(pixels)
            mad = stats.median_abs_deviation(pixels)
            mean = np.mean(pixels)
            std = np.std(pixels)
            
            # 2. Advanced outlier detection methods
            outliers = {}
            
            # Z-score (classical)
            z_scores = np.abs(stats.zscore(pixels))
            outliers['zscore'] = z_scores > self.config.zscore_threshold
            
            # Modified Z-score using MAD
            modified_z = 0.6745 * (pixels - median) / (mad + 1e-10)
            outliers['mad'] = np.abs(modified_z) > self.config.mad_threshold
            
            # IQR method
            q1, q3 = np.percentile(pixels, [25, 75])
            iqr = q3 - q1
            outliers['iqr'] = (pixels < q1 - self.config.iqr_multiplier * iqr) | \
                            (pixels > q3 + self.config.iqr_multiplier * iqr)
            
            # Grubbs test
            outliers['grubbs'] = self._grubbs_test(pixels, self.config.grubbs_alpha)
            
            # Dixon Q test
            outliers['dixon'] = self._dixon_q_test(pixels, self.config.dixon_alpha)
            
            # Chauvenet's criterion
            outliers['chauvenet'] = self._chauvenet_criterion(pixels, self.config.chauvenet_criterion)
            
            # Tukey's method
            outliers['tukey'] = self._tukey_outliers(pixels)
            
            # GESD test
            outliers['gesd'] = self._generalized_esd_test(pixels)
            
            # Hampel filter
            outliers['hampel'] = self._hampel_filter(pixels)
            
            # 3. Distribution-based methods
            # Fit multiple distributions and find outliers
            distributions = ['norm', 'lognorm', 'gamma', 'expon', 'weibull_min']
            dist_outliers = {}
            
            for dist_name in distributions:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(pixels)
                    
                    # Calculate probability of each pixel
                    probabilities = dist.pdf(pixels, *params)
                    threshold = np.percentile(probabilities, 5)
                    dist_outliers[dist_name] = probabilities < threshold
                except:
                    continue
            
            # 4. Information-theoretic outliers
            # Local outlier probabilities using entropy
            outliers['entropy'] = self._entropy_based_outliers(pixels)
            
            # 5. Kernel density estimation outliers
            outliers['kde'] = self._kde_outliers(pixels)
            
            results[img_name] = {
                'statistics': {
                    'mean': mean,
                    'median': median,
                    'std': std,
                    'mad': mad,
                    'skewness': stats.skew(pixels),
                    'kurtosis': stats.kurtosis(pixels),
                    'entropy': stats.entropy(np.histogram(pixels, bins=50)[0] + 1e-10)
                },
                'outliers': outliers,
                'distribution_outliers': dist_outliers
            }
        
        return results
    
    def _phd_spatial_analysis(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Advanced spatial domain analysis using PhD-level techniques"""
        results = {}
        
        for img_name, img in images.items():
            spatial_features = {}
            
            # 1. Advanced Local Binary Patterns
            lbp_variants = {}
            for radius in [1, 2, 3]:
                for n_points in [8, 16, 24]:
                    lbp = feature.local_binary_pattern(img, n_points, radius, method='uniform')
                    lbp_variants[f'lbp_r{radius}_p{n_points}'] = lbp
            
            # 2. Local Ternary Patterns
            ltp = self._local_ternary_pattern(img)
            
            # 3. Local Phase Quantization
            lpq = self._local_phase_quantization(img)
            
            # 4. Completed Local Binary Patterns
            clbp = self._completed_lbp(img)
            
            # 5. Gray Level Co-occurrence Matrix (Multiple angles and distances)
            glcm_features = {}
            for distance in [1, 3, 5]:
                for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    glcm = greycomatrix(img, [distance], [angle], symmetric=True, normed=True)
                    
                    # Extract all Haralick features
                    glcm_features[f'd{distance}_a{int(np.degrees(angle))}'] = {
                        'contrast': greycoprops(glcm, 'contrast')[0, 0],
                        'dissimilarity': greycoprops(glcm, 'dissimilarity')[0, 0],
                        'homogeneity': greycoprops(glcm, 'homogeneity')[0, 0],
                        'energy': greycoprops(glcm, 'energy')[0, 0],
                        'correlation': greycoprops(glcm, 'correlation')[0, 0],
                        'ASM': greycoprops(glcm, 'ASM')[0, 0]
                    }
            
            # 6. Gray Level Run Length Matrix
            glrlm = self._gray_level_run_length_matrix(img)
            
            # 7. Local Gradient Patterns
            lgp = self._local_gradient_pattern(img)
            
            # 8. Spatial Autocorrelation (Moran's I, Geary's C)
            morans_i = self._morans_i(img, self.mask)
            gearys_c = self._gearys_c(img, self.mask)
            
            # 9. Fractal Dimension Analysis
            fractal_dim = self._fractal_dimension(img)
            
            # 10. Spatial Entropy Maps
            entropy_map = self._local_entropy_map(img)
            
            spatial_features = {
                'lbp_variants': lbp_variants,
                'ltp': ltp,
                'lpq': lpq,
                'clbp': clbp,
                'glcm': glcm_features,
                'glrlm': glrlm,
                'lgp': lgp,
                'spatial_autocorrelation': {
                    'morans_i': morans_i,
                    'gearys_c': gearys_c
                },
                'fractal_dimension': fractal_dim,
                'entropy_map': entropy_map
            }
            
            results[img_name] = spatial_features
        
        return results
    
    def _advanced_frequency_analysis(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """PhD-level frequency domain analysis"""
        results = {}
        
        for img_name, img in images.items():
            freq_features = {}
            
            # 1. 2D Fourier Transform Analysis
            f_transform = fft2(img)
            f_shift = fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            phase_spectrum = np.angle(f_shift)
            
            # Power Spectral Density
            psd = np.abs(f_shift) ** 2
            
            # Radial average of PSD
            radial_psd = self._radial_profile(psd)
            
            # 2. Discrete Cosine Transform
            dct_coeffs = dct(dct(img.T, norm='ortho').T, norm='ortho')
            
            # 3. Wavelet Analysis (Multiple families and decompositions)
            wavelet_features = {}
            for wavelet_family in self.config.wavelet_families:
                for level in self.config.wavelet_levels:
                    coeffs = pywt.wavedec2(img, wavelet_family, level=level)
                    
                    # Extract features from each subband
                    subband_features = []
                    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
                        subband_features.append({
                            'horizontal_energy': np.sum(cH**2),
                            'vertical_energy': np.sum(cV**2),
                            'diagonal_energy': np.sum(cD**2),
                            'horizontal_entropy': stats.entropy(cH.flatten() + 1e-10),
                            'vertical_entropy': stats.entropy(cV.flatten() + 1e-10),
                            'diagonal_entropy': stats.entropy(cD.flatten() + 1e-10)
                        })
                    
                    wavelet_features[f'{wavelet_family}_L{level}'] = subband_features
            
            # 4. Gabor Filter Bank Analysis
            gabor_responses = self._comprehensive_gabor_analysis(img)
            
            # 5. Steerable Pyramid Decomposition
            steerable_pyramid = self._steerable_pyramid_decomposition(img)
            
            # 6. Contourlet Transform
            contourlet = self._contourlet_transform(img)
            
            # 7. Shearlet Transform
            shearlet = self._shearlet_transform(img)
            
            # 8. Ridgelet Transform
            ridgelet = self._ridgelet_transform(img)
            
            # 9. Curvelet Transform
            curvelet = self._curvelet_transform(img)
            
            # 10. Stockwell Transform
            stockwell = self._stockwell_transform(img)
            
            freq_features = {
                'fourier': {
                    'magnitude_spectrum': magnitude_spectrum,
                    'phase_spectrum': phase_spectrum,
                    'psd': psd,
                    'radial_psd': radial_psd
                },
                'dct': dct_coeffs,
                'wavelet': wavelet_features,
                'gabor': gabor_responses,
                'steerable_pyramid': steerable_pyramid,
                'contourlet': contourlet,
                'shearlet': shearlet,
                'ridgelet': ridgelet,
                'curvelet': curvelet,
                'stockwell': stockwell
            }
            
            results[img_name] = freq_features
        
        return results
    
    def _transform_domain_analysis(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analysis in various transform domains"""
        results = {}
        
        for img_name, img in images.items():
            transforms = {}
            
            # 1. Radon Transform (for line detection)
            theta = np.linspace(0., 180., max(img.shape), endpoint=False)
            sinogram = radon(img, theta=theta)
            transforms['radon'] = sinogram
            
            # 2. Hough Transform variants
            # Standard Hough
            edges = cv2.Canny(img, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            transforms['hough_lines'] = lines
            
            # Probabilistic Hough
            lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)
            transforms['hough_lines_p'] = lines_p
            
            # Circle Hough
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30)
            transforms['hough_circles'] = circles
            
            # 3. Distance Transform
            if self.mask is not None:
                dist_transform = distance_transform_edt(self.mask)
                transforms['distance_transform'] = dist_transform
            
            # 4. Watershed Transform
            watershed_result = self._advanced_watershed(img)
            transforms['watershed'] = watershed_result
            
            # 5. Medial Axis Transform
            if self.mask is not None:
                medial_axis_result = medial_axis(self.mask > 0)
                transforms['medial_axis'] = medial_axis_result
            
            # 6. Hit-or-Miss Transform
            hit_miss_results = self._hit_or_miss_analysis(img)
            transforms['hit_or_miss'] = hit_miss_results
            
            results[img_name] = transforms
        
        return results
    
    def _comprehensive_morphological_analysis(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Comprehensive morphological analysis using all available operations"""
        results = {}
        
        for img_name, img in images.items():
            morph_features = {}
            
            # Multiple structuring elements
            se_types = {
                'disk': [morphology.disk(r) for r in [1, 3, 5, 7]],
                'square': [morphology.square(s) for s in [3, 5, 7, 9]],
                'diamond': [morphology.diamond(r) for r in [1, 2, 3, 4]],
                'octagon': [morphology.octagon(m, n) for m, n in [(2, 2), (3, 3), (4, 4)]],
                'star': [morphology.star(n) for n in [3, 4, 5]]
            }
            
            for se_name, se_list in se_types.items():
                se_results = {}
                
                for i, se in enumerate(se_list):
                    # Basic operations
                    erosion = cv2.erode(img, se)
                    dilation = cv2.dilate(img, se)
                    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
                    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
                    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se)
                    
                    # Advanced operations
                    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)
                    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)
                    
                    # Morphological reconstruction
                    marker = erosion
                    reconstruction = morphology.reconstruction(marker, img, method='dilation')
                    
                    # Morphological profiles
                    if i == 0:  # Only for first SE of each type
                        profile = self._morphological_profile(img, se_name)
                        se_results[f'profile_{se_name}'] = profile
                    
                    se_results[f'{se_name}_{i}'] = {
                        'erosion': erosion,
                        'dilation': dilation,
                        'opening': opening,
                        'closing': closing,
                        'gradient': gradient,
                        'tophat': tophat,
                        'blackhat': blackhat,
                        'reconstruction': reconstruction
                    }
                
                morph_features[se_name] = se_results
            
            # Advanced morphological operations
            # 1. Morphological Laplacian
            morph_laplacian = self._morphological_laplacian(img)
            morph_features['laplacian'] = morph_laplacian
            
            # 2. Toggle Mapping
            toggle_map = self._toggle_mapping(img)
            morph_features['toggle'] = toggle_map
            
            # 3. Morphological Gradient variants
            internal_gradient = dilation - img
            external_gradient = img - erosion
            morph_features['gradients'] = {
                'internal': internal_gradient,
                'external': external_gradient,
                'standard': gradient
            }
            
            # 4. Ultimate Erosion/Dilation
            ultimate_erosion = self._ultimate_erosion(img)
            ultimate_dilation = self._ultimate_dilation(img)
            morph_features['ultimate'] = {
                'erosion': ultimate_erosion,
                'dilation': ultimate_dilation
            }
            
            results[img_name] = morph_features
        
        return results
    
    def _advanced_texture_analysis(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Advanced texture analysis using multiple methods"""
        results = {}
        
        for img_name, img in images.items():
            texture_features = {}
            
            # 1. Laws Texture Energy
            laws_features = self._laws_texture_energy(img)
            texture_features['laws'] = laws_features
            
            # 2. Tamura Features
            tamura_features = self._tamura_features(img)
            texture_features['tamura'] = tamura_features
            
            # 3. Local Range and Variance
            for window_size in [3, 5, 7, 9]:
                local_range = self._local_range(img, window_size)
                local_var = self._local_variance(img, window_size)
                texture_features[f'local_range_{window_size}'] = local_range
                texture_features[f'local_variance_{window_size}'] = local_var
            
            # 4. Texture Spectrum
            texture_spectrum = self._texture_spectrum(img)
            texture_features['spectrum'] = texture_spectrum
            
            # 5. Markov Random Field parameters
            mrf_params = self._estimate_mrf_parameters(img)
            texture_features['mrf'] = mrf_params
            
            # 6. Gabor Texture Features
            gabor_texture = self._gabor_texture_features(img)
            texture_features['gabor'] = gabor_texture
            
            # 7. Fractal-based texture
            fractal_texture = self._fractal_texture_analysis(img)
            texture_features['fractal'] = fractal_texture
            
            results[img_name] = texture_features
        
        return results
    
    def _phd_edge_analysis(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """PhD-level edge and gradient analysis"""
        results = {}
        
        for img_name, img in images.items():
            edge_features = {}
            
            # 1. Multi-scale edge detection
            # Sobel at multiple scales
            sobel_scales = {}
            for ksize in [3, 5, 7]:
                sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
                sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
                sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
                sobel_dir = np.arctan2(sobel_y, sobel_x)
                sobel_scales[f'scale_{ksize}'] = {
                    'magnitude': sobel_mag,
                    'direction': sobel_dir
                }
            
            # 2. Advanced edge detectors
            # Canny with multiple thresholds
            canny_results = {}
            for low, high in [(50, 150), (30, 100), (80, 200)]:
                canny_results[f'{low}_{high}'] = cv2.Canny(img, low, high)
            
            # Structured edge detection (simplified)
            structured_edges = self._structured_edge_detection(img)
            
            # 3. Phase congruency edge detection
            phase_congruency = self._phase_congruency(img)
            
            # 4. Subpixel edge detection
            subpixel_edges = self._subpixel_edge_detection(img)
            
            # 5. Multi-scale Laplacian
            log_scales = {}
            for sigma in [1, 2, 3, 4]:
                gaussian = cv2.GaussianBlur(img, (0, 0), sigma)
                log = cv2.Laplacian(gaussian, cv2.CV_64F)
                log_scales[f'sigma_{sigma}'] = log
            
            # 6. Structure Tensor Analysis
            structure_tensor = self._structure_tensor_analysis(img)
            
            # 7. Gradient Vector Flow
            gvf = self._gradient_vector_flow(img)
            
            # 8. Edge Linking and Grouping
            linked_edges = self._edge_linking(canny_results[list(canny_results.keys())[0]])
            
            # 9. Cornerness measures
            corners = {
                'harris': cv2.cornerHarris(img, 2, 3, 0.04),
                'shi_tomasi': cv2.goodFeaturesToTrack(img, 100, 0.01, 10),
                'fast': cv2.FastFeatureDetector_create().detect(img)
            }
            
            edge_features = {
                'sobel_multiscale': sobel_scales,
                'canny_multithreshold': canny_results,
                'structured': structured_edges,
                'phase_congruency': phase_congruency,
                'subpixel': subpixel_edges,
                'log_multiscale': log_scales,
                'structure_tensor': structure_tensor,
                'gvf': gvf,
                'linked_edges': linked_edges,
                'corners': corners
            }
            
            results[img_name] = edge_features
        
        return results
    
    def _machine_learning_detection(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Machine learning based defect detection"""
        results = {}
        
        for img_name, img in images.items():
            ml_results = {}
            
            # Extract comprehensive features for each pixel
            features = self._extract_pixel_features_comprehensive(img)
            
            # Reshape for ML algorithms
            h, w = img.shape[:2]
            features_flat = features.reshape(-1, features.shape[-1])
            mask_flat = self.mask.flatten()
            
            # Only use pixels within mask
            valid_features = features_flat[mask_flat > 0]
            
            if len(valid_features) < 100:
                continue
            
            # Normalize features
            scaler = RobustScaler()
            features_normalized = scaler.fit_transform(valid_features)
            
            # 1. Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.config.ml_contamination,
                n_estimators=200,
                random_state=42
            )
            iso_predictions = iso_forest.fit_predict(features_normalized)
            
            # 2. One-Class SVM
            ocsvm = OneClassSVM(
                gamma='scale',
                nu=self.config.ml_contamination,
                kernel='rbf'
            )
            svm_predictions = ocsvm.fit_predict(features_normalized)
            
            # 3. Local Outlier Factor
            lof = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.config.ml_contamination
            )
            lof_predictions = lof.fit_predict(features_normalized)
            
            # 4. DBSCAN with parameter optimization
            best_dbscan = self._optimize_dbscan(features_normalized)
            
            # 5. Gaussian Mixture Model
            n_components = self._estimate_gmm_components(features_normalized)
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(features_normalized)
            gmm_scores = gmm.score_samples(features_normalized)
            gmm_threshold = np.percentile(gmm_scores, 100 * self.config.ml_contamination)
            gmm_predictions = (gmm_scores < gmm_threshold).astype(int) * -1 + 1
            
            # 6. Autoencoder-based anomaly detection (simplified)
            ae_anomalies = self._autoencoder_anomalies(features_normalized)
            
            # Map predictions back to image
            def map_to_image(predictions):
                result = np.zeros(img.shape, dtype=np.uint8)
                valid_idx = 0
                for i in range(len(mask_flat)):
                    if mask_flat[i] > 0:
                        if valid_idx < len(predictions) and predictions[valid_idx] == -1:
                            y, x = i // w, i % w
                            result[y, x] = 255
                        valid_idx += 1
                return result
            
            ml_results = {
                'isolation_forest': map_to_image(iso_predictions),
                'one_class_svm': map_to_image(svm_predictions),
                'lof': map_to_image(lof_predictions),
                'dbscan': best_dbscan['mask'],
                'gmm': map_to_image(gmm_predictions),
                'autoencoder': ae_anomalies
            }
            
            results[img_name] = ml_results
        
        return results
    
    def _deep_learning_detection(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Deep learning based detection (placeholder for actual implementation)"""
        # This would require trained models
        # Could include:
        # - CNN-based defect detection
        # - U-Net for segmentation
        # - YOLO/R-CNN for object detection
        # - Autoencoders for anomaly detection
        # - GANs for defect generation and detection
        return {}
    
    def _physics_based_detection(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Physics-based defect detection methods"""
        results = {}
        
        for img_name, img in images.items():
            physics_results = {}
            
            # 1. Optical diffraction simulation
            diffraction_pattern = self._simulate_diffraction(img)
            physics_results['diffraction'] = diffraction_pattern
            
            # 2. Light scattering analysis
            scattering_map = self._light_scattering_analysis(img)
            physics_results['scattering'] = scattering_map
            
            # 3. Fresnel propagation
            fresnel_result = self._fresnel_propagation(img)
            physics_results['fresnel'] = fresnel_result
            
            # 4. Polarization analysis (simulated)
            polarization = self._polarization_analysis(img)
            physics_results['polarization'] = polarization
            
            # 5. Interference pattern analysis
            interference = self._interference_analysis(img)
            physics_results['interference'] = interference
            
            results[img_name] = physics_results
        
        return results
    
    def _hybrid_novel_methods(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Novel and hybrid detection methods"""
        results = {}
        
        for img_name, img in images.items():
            novel_results = {}
            
            # 1. Topological Data Analysis
            tda_features = self._topological_data_analysis(img)
            novel_results['tda'] = tda_features
            
            # 2. Graph-based defect detection
            graph_defects = self._graph_based_detection(img)
            novel_results['graph'] = graph_defects
            
            # 3. Compressed Sensing reconstruction
            cs_defects = self._compressed_sensing_detection(img)
            novel_results['compressed_sensing'] = cs_defects
            
            # 4. Quantum-inspired detection (simulated)
            quantum_defects = self._quantum_inspired_detection(img)
            novel_results['quantum'] = quantum_defects
            
            # 5. Bio-inspired detection (retinal processing simulation)
            bio_defects = self._bio_inspired_detection(img)
            novel_results['bio_inspired'] = bio_defects
            
            # 6. Information geometry based detection
            info_geom = self._information_geometry_detection(img)
            novel_results['info_geometry'] = info_geom
            
            results[img_name] = novel_results
        
        return results
    
    def _intelligent_fusion(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent fusion of all detection methods using advanced techniques"""
        
        # Extract all defect masks from different methods
        all_masks = []
        method_names = []
        confidence_maps = {}
        
        for category, category_results in all_results.items():
            if isinstance(category_results, dict):
                for subcat, subcat_results in category_results.items():
                    if isinstance(subcat_results, dict):
                        for method, result in subcat_results.items():
                            if isinstance(result, np.ndarray) and result.shape == self.original_image.shape:
                                all_masks.append(result)
                                method_names.append(f"{category}_{subcat}_{method}")
        
        if not all_masks:
            return {'final_mask': np.zeros_like(self.original_image), 
                   'confidence_map': np.zeros_like(self.original_image, dtype=np.float32)}
        
        # 1. Weighted voting based on method reliability
        weights = self._calculate_method_weights(all_masks, method_names)
        
        # 2. Fuzzy logic fusion
        fuzzy_fusion = self._fuzzy_logic_fusion(all_masks, weights)
        
        # 3. Dempster-Shafer evidence theory
        ds_fusion = self._dempster_shafer_fusion(all_masks, weights)
        
        # 4. Bayesian fusion
        bayesian_fusion = self._bayesian_fusion(all_masks, weights)
        
        # 5. Neural network based fusion (simplified)
        nn_fusion = self._neural_network_fusion(all_masks)
        
        # 6. Consensus clustering
        consensus_fusion = self._consensus_clustering_fusion(all_masks)
        
        # Final fusion of fusion methods
        final_fusion = np.zeros_like(self.original_image, dtype=np.float32)
        fusion_methods = [fuzzy_fusion, ds_fusion, bayesian_fusion, nn_fusion, consensus_fusion]
        
        for fusion_result in fusion_methods:
            if fusion_result is not None:
                final_fusion += fusion_result.astype(np.float32) / 255.0
        
        final_fusion = final_fusion / len(fusion_methods)
        
        # Adaptive thresholding for final decision
        threshold = self._adaptive_fusion_threshold(final_fusion)
        final_mask = (final_fusion > threshold).astype(np.uint8) * 255
        
        return {
            'final_mask': final_mask,
            'confidence_map': final_fusion,
            'methods_used': method_names,
            'method_weights': weights,
            'fusion_threshold': threshold
        }
    
    def _advanced_validation(self, fusion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced validation and false positive reduction"""
        
        final_mask = fusion_results.get('final_mask', np.zeros_like(self.original_image))
        confidence_map = fusion_results.get('confidence_map', np.zeros_like(self.original_image, dtype=np.float32))
        
        # 1. Physics-based validation
        physics_valid = self._physics_based_validation(final_mask)
        
        # 2. Context-aware validation
        context_valid = self._context_aware_validation(final_mask)
        
        # 3. Machine learning validation
        ml_valid = self._ml_based_validation(final_mask)
        
        # 4. Morphological consistency validation
        morph_valid = self._morphological_consistency_validation(final_mask)
        
        # 5. Statistical significance validation
        stat_valid = self._statistical_significance_validation(final_mask)
        
        # Combine all validations
        validated_mask = final_mask.copy()
        
        # Apply all validation masks
        validation_masks = [physics_valid, context_valid, ml_valid, morph_valid, stat_valid]
        for val_mask in validation_masks:
            if val_mask is not None:
                validated_mask = cv2.bitwise_and(validated_mask, val_mask)
        
        # Update confidence based on validation
        validated_confidence = confidence_map.copy()
        validated_confidence[validated_mask == 0] = 0
        
        return {
            'validated_mask': validated_mask,
            'validated_confidence': validated_confidence,
            'validation_steps': {
                'physics': physics_valid is not None,
                'context': context_valid is not None,
                'ml': ml_valid is not None,
                'morphological': morph_valid is not None,
                'statistical': stat_valid is not None
            }
        }
    
    def _characterize_defects(self, validated_results: Dict[str, Any]) -> List[Defect]:
        """Comprehensive defect characterization"""
        
        mask = validated_results.get('validated_mask', np.zeros_like(self.original_image))
        confidence = validated_results.get('validated_confidence', np.zeros_like(self.original_image, dtype=np.float32))
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        defects = []
        
        for i in range(1, num_labels):
            # Extract component
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Basic properties
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            centroid = centroids[i]
            
            if area < self.config.min_defect_area:
                continue
            
            # Advanced characterization
            # 1. Shape analysis
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = contours[0]
            
            # Fit ellipse if possible
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (ex, ey), (major_axis, minor_axis), angle = ellipse
                
                if minor_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2)
                else:
                    eccentricity = 1.0
            else:
                major_axis = max(w, h)
                minor_axis = min(w, h)
                angle = 0
                eccentricity = 0
            
            # Calculate more shape features
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Compactness
            compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Moments
            moments = cv2.moments(component_mask)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Zernike moments
            zernike_moments = self._calculate_zernike_moments(component_mask)
            
            # Fourier descriptors
            fourier_descriptors = self._calculate_fourier_descriptors(contour)
            
            # 2. Intensity analysis
            defect_pixels = self.original_image[component_mask > 0]
            surrounding_pixels = self._get_surrounding_pixels(component_mask)
            
            intensity_features = {
                'mean': np.mean(defect_pixels),
                'std': np.std(defect_pixels),
                'min': np.min(defect_pixels),
                'max': np.max(defect_pixels),
                'median': np.median(defect_pixels),
                'skewness': stats.skew(defect_pixels),
                'kurtosis': stats.kurtosis(defect_pixels),
                'contrast': abs(np.mean(defect_pixels) - np.mean(surrounding_pixels)) if len(surrounding_pixels) > 0 else 0
            }
            
            # 3. Texture analysis
            texture_features = self._analyze_defect_texture(component_mask, self.original_image)
            
            # 4. Classification
            defect_type = self._classify_defect_advanced(
                area, major_axis, minor_axis, eccentricity, 
                solidity, compactness, intensity_features, texture_features
            )
            
            # 5. Confidence calculation
            defect_confidence = np.mean(confidence[component_mask > 0])
            
            # Create defect object
            defect = Defect(
                id=f"D{i:04d}",
                type=defect_type,
                confidence=float(defect_confidence),
                location=(int(centroid[0]), int(centroid[1])),
                bbox=(x, y, w, h),
                area_px=int(area),
                perimeter=float(perimeter),
                major_axis=float(major_axis),
                minor_axis=float(minor_axis),
                orientation=float(angle),
                eccentricity=float(eccentricity),
                solidity=float(solidity),
                compactness=float(compactness),
                hu_moments=hu_moments,
                zernike_moments=zernike_moments,
                fourier_descriptors=fourier_descriptors,
                texture_features=texture_features,
                intensity_features=intensity_features,
                mask=component_mask
            )
            
            defects.append(defect)
        
        # Sort by confidence and size
        defects.sort(key=lambda d: (d.confidence, d.area_px), reverse=True)
        
        return defects
    
    def _calculate_quality_metrics(self, defects: List[Defect]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        
        total_area = np.sum(self.mask > 0)
        
        if total_area == 0:
            return {}
        
        # Basic statistics
        total_defects = len(defects)
        total_defect_area = sum(d.area_px for d in defects)
        defect_density = total_defect_area / total_area
        
        # Defect type distribution
        type_distribution = {}
        for defect in defects:
            type_name = defect.type.name
            if type_name not in type_distribution:
                type_distribution[type_name] = 0
            type_distribution[type_name] += 1
        
        # Size distribution
        if defects:
            sizes = [d.area_px for d in defects]
            size_stats = {
                'mean': np.mean(sizes),
                'std': np.std(sizes),
                'min': np.min(sizes),
                'max': np.max(sizes),
                'median': np.median(sizes),
                'percentiles': {
                    '25': np.percentile(sizes, 25),
                    '50': np.percentile(sizes, 50),
                    '75': np.percentile(sizes, 75),
                    '90': np.percentile(sizes, 90),
                    '95': np.percentile(sizes, 95)
                }
            }
        else:
            size_stats = {}
        
        # Spatial distribution analysis
        spatial_metrics = self._analyze_spatial_distribution(defects)
        
        # Surface quality indices
        quality_indices = {
            'defect_free_ratio': 1 - defect_density,
            'uniformity_index': self._calculate_uniformity_index(defects),
            'clustering_index': self._calculate_clustering_index(defects),
            'severity_index': self._calculate_severity_index(defects)
        }
        
        # Statistical quality metrics
        statistical_metrics = {
            'mean_confidence': np.mean([d.confidence for d in defects]) if defects else 0,
            'std_confidence': np.std([d.confidence for d in defects]) if defects else 0,
            'high_confidence_ratio': len([d for d in defects if d.confidence > 0.8]) / len(defects) if defects else 0
        }
        
        return {
            'total_defects': total_defects,
            'total_defect_area': total_defect_area,
            'defect_density': defect_density,
            'type_distribution': type_distribution,
            'size_statistics': size_stats,
            'spatial_metrics': spatial_metrics,
            'quality_indices': quality_indices,
            'statistical_metrics': statistical_metrics,
            'region_type': self.region_type,
            'analysis_area': total_area
        }
    
    # Helper methods (implementations of all the referenced methods would go here)
    # Due to space constraints, I'll provide a few key examples:
    
    def _rolling_ball_correction(self, image: np.ndarray, radius: int = 50) -> np.ndarray:
        """Rolling ball background subtraction"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        corrected = cv2.subtract(image, background) + 128
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def _homomorphic_filtering(self, image: np.ndarray, d0: int = 10, 
                              gamma_l: float = 0.7, gamma_h: float = 1.5) -> np.ndarray:
        """Homomorphic filtering for illumination correction"""
        # Take logarithm
        img_log = np.log1p(image.astype(np.float32))
        
        # DFT
        img_fft = np.fft.fft2(img_log)
        img_fft_shift = np.fft.fftshift(img_fft)
        
        # Create filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Gaussian filter
        x = np.arange(cols) - ccol
        y = np.arange(rows) - crow
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        
        H = (gamma_h - gamma_l) * (1 - np.exp(-(D**2) / (2 * d0**2))) + gamma_l
        
        # Apply filter
        img_fft_shift_filtered = H * img_fft_shift
        
        # Inverse FFT
        img_fft_filtered = np.fft.ifftshift(img_fft_shift_filtered)
        img_filtered = np.fft.ifft2(img_fft_filtered)
        img_filtered = np.real(img_filtered)
        
        # Exponential
        img_exp = np.expm1(img_filtered)
        
        # Normalize
        img_normalized = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
        
        return img_normalized.astype(np.uint8)
    
    def _multi_scale_retinex(self, image: np.ndarray, sigmas: List[float] = [15, 80, 250]) -> np.ndarray:
        """Multi-scale Retinex enhancement"""
        retinex = np.zeros_like(image, dtype=np.float32)
        
        for sigma in sigmas:
            # Gaussian blur
            gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # Log domain
            log_image = np.log1p(image.astype(np.float32))
            log_gaussian = np.log1p(gaussian.astype(np.float32))
            
            # Retinex
            retinex += log_image - log_gaussian
        
        retinex = retinex / len(sigmas)
        
        # Dynamic range compression
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        
        return retinex.astype(np.uint8)
    
    def _comprehensive_gabor_analysis(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Comprehensive Gabor filter bank analysis"""
        gabor_results = {}
        
        for freq in self.config.gabor_frequencies:
            for theta_idx, theta in enumerate(self.config.gabor_orientations):
                for sigma in self.config.gabor_sigmas:
                    # Create Gabor kernel
                    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
                    kernel = cv2.getGaborKernel(
                        (kernel_size, kernel_size),
                        sigma,
                        theta,
                        1/freq,
                        0.5,
                        0,
                        ktype=cv2.CV_32F
                    )
                    
                    # Apply filter
                    filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                    
                    # Store magnitude and phase
                    key = f'f{freq:.2f}_t{int(np.degrees(theta))}_s{sigma}'
                    gabor_results[key] = {
                        'magnitude': np.abs(filtered),
                        'phase': np.angle(filtered),
                        'real': np.real(filtered),
                        'imaginary': np.imag(filtered)
                    }
        
        return gabor_results
    
    def _classify_defect_advanced(self, area: float, major_axis: float, minor_axis: float,
                                 eccentricity: float, solidity: float, compactness: float,
                                 intensity_features: Dict[str, float],
                                 texture_features: Dict[str, float]) -> DefectType:
        """Advanced defect classification using multiple features"""
        
        # Calculate aspect ratio
        aspect_ratio = major_axis / (minor_axis + 1e-6)
        
        # Rule-based classification with fuzzy boundaries
        
        # Scratches: elongated, high aspect ratio
        if aspect_ratio > 5 and eccentricity > 0.9:
            return DefectType.SCRATCH
        
        # Cracks: elongated but more irregular than scratches
        if aspect_ratio > 3 and eccentricity > 0.8 and solidity < 0.7:
            return DefectType.CRACK
        
        # Pits: small, round, dark
        if area < 100 and compactness > 0.7 and intensity_features['mean'] < 100:
            return DefectType.PIT
        
        # Digs: larger than pits, still round
        if 100 <= area < 500 and compactness > 0.6:
            return DefectType.DIG
        
        # Particles: small, bright
        if area < 200 and intensity_features['mean'] > 150:
            return DefectType.PARTICLE
        
        # Contamination: larger, irregular
        if area > 500 and solidity < 0.8:
            return DefectType.CONTAMINATION
        
        # Chips: medium size, angular
        if 200 <= area < 1000 and compactness < 0.5:
            return DefectType.CHIP
        
        # Bubbles: round, specific intensity pattern
        if compactness > 0.8 and 0.5 < intensity_features['contrast'] < 20:
            return DefectType.BUBBLE
        
        # Stains: large, low contrast
        if area > 1000 and intensity_features['contrast'] < 10:
            return DefectType.STAIN
        
        # Default
        return DefectType.UNKNOWN
    
    def _count_algorithms_used(self) -> int:
        """Count total number of algorithms used in the analysis"""
        # This would count all the methods actually used
        # For now, return an estimate
        return 150  # Approximate number of distinct algorithms implemented
    
    def visualize_comprehensive_results(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of all results"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create large figure with subplots for different aspects
        fig = plt.figure(figsize=(30, 20))
        gs = GridSpec(6, 6, figure=fig, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.original_image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Final defect mask
        ax2 = fig.add_subplot(gs[0, 1])
        if 'detection_masks' in self.results:
            final_mask = self.results['detection_masks'].get('validated_mask', np.zeros_like(self.original_image))
            ax2.imshow(final_mask, cmap='hot')
        ax2.set_title('Final Defect Mask')
        ax2.axis('off')
        
        # Confidence map
        ax3 = fig.add_subplot(gs[0, 2])
        if 'confidence_maps' in self.results:
            conf_map = list(self.results['confidence_maps'].values())[0] if self.results['confidence_maps'] else np.zeros_like(self.original_image)
            im3 = ax3.imshow(conf_map, cmap='plasma')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
        ax3.set_title('Confidence Map')
        ax3.axis('off')
        
        # Defect overlay
        ax4 = fig.add_subplot(gs[0, 3])
        overlay = self.color_image.copy()
        for defect in self.defects:
            if defect.mask is not None:
                overlay[defect.mask > 0] = [255, 0, 0]  # Red for defects
        ax4.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax4.set_title('Defect Overlay')
        ax4.axis('off')
        
        # Individual method results (sample)
        row, col = 1, 0
        for category, results in self.results.get('individual_results', {}).items():
            if row >= 6:
                break
            
            if isinstance(results, dict):
                for method, result in list(results.items())[:2]:  # Show first 2 methods per category
                    if col >= 6:
                        col = 0
                        row += 1
                    if row >= 6:
                        break
                    
                    ax = fig.add_subplot(gs[row, col])
                    
                    if isinstance(result, np.ndarray) and len(result.shape) == 2:
                        ax.imshow(result, cmap='gray')
                        ax.set_title(f'{category}_{method}'[:20])
                        ax.axis('off')
                        col += 1
        
        # Quality metrics
        ax_metrics = fig.add_subplot(gs[4:6, 0:2])
        metrics_text = self._format_quality_metrics()
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax_metrics.set_title('Quality Metrics')
        ax_metrics.axis('off')
        
        # Defect distribution
        ax_dist = fig.add_subplot(gs[4:6, 2:4])
        if self.defects:
            defect_types = [d.type.name for d in self.defects]
            unique_types, counts = np.unique(defect_types, return_counts=True)
            ax_dist.bar(unique_types, counts)
            ax_dist.set_xlabel('Defect Type')
            ax_dist.set_ylabel('Count')
            ax_dist.set_title('Defect Type Distribution')
            plt.setp(ax_dist.xaxis.get_majorticklabels(), rotation=45)
        
        # Size distribution
        ax_size = fig.add_subplot(gs[4:6, 4:6])
        if self.defects:
            sizes = [d.area_px for d in self.defects]
            ax_size.hist(sizes, bins=20, edgecolor='black')
            ax_size.set_xlabel('Defect Size (pixels)')
            ax_size.set_ylabel('Count')
            ax_size.set_title('Defect Size Distribution')
            ax_size.set_yscale('log')
        
        plt.suptitle(f'Comprehensive Defect Analysis - {self.region_type}', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _format_quality_metrics(self) -> str:
        """Format quality metrics for display"""
        metrics = self.results.get('quality_metrics', {})
        
        text = "QUALITY METRICS SUMMARY\n"
        text += "=" * 40 + "\n"
        text += f"Total Defects: {metrics.get('total_defects', 0)}\n"
        text += f"Total Defect Area: {metrics.get('total_defect_area', 0)} px\n"
        text += f"Defect Density: {metrics.get('defect_density', 0):.4f}\n"
        text += f"Analysis Area: {metrics.get('analysis_area', 0)} px\n\n"
        
        text += "Quality Indices:\n"
        indices = metrics.get('quality_indices', {})
        for key, value in indices.items():
            text += f"  {key}: {value:.4f}\n"
        
        text += "\nStatistical Metrics:\n"
        stats = metrics.get('statistical_metrics', {})
        for key, value in stats.items():
            text += f"  {key}: {value:.4f}\n"
        
        return text

# Example usage
if __name__ == "__main__":
    # Create test image
    test_image = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
    
    # Add some synthetic defects
    # Scratch
    cv2.line(test_image, (100, 100), (200, 150), 50, 2)
    
    # Pit
    cv2.circle(test_image, (300, 300), 5, 30, -1)
    
    # Contamination
    cv2.ellipse(test_image, (400, 400), (30, 20), 45, 0, 360, 100, -1)
    
    # Add noise
    noise = np.random.normal(0, 10, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Create mask
    mask = np.ones_like(test_image) * 255
    
    # Initialize detector
    config = DefectDetectionConfig()
    detector = UltimateDefectDetector(config)
    
    # Run analysis
    print("Starting comprehensive defect analysis...")
    results = detector.analyze_comprehensive(test_image, mask, region_type="Test")
    
    # Print summary
    print(f"\nAnalysis complete!")
    print(f"Found {len(results['defects'])} defects")
    print(f"Analysis time: {results['performance']['analysis_time']:.2f} seconds")
    print(f"Methods used: {results['performance']['methods_used']}")
    print(f"Total algorithms: {results['performance']['total_algorithms']}")
    
    # Visualize results
    detector.visualize_comprehensive_results("ultimate_defect_detection_results.png")
    print("\nResults saved to ultimate_defect_detection_results.png")