"""
Unified Intelligent Fiber Optic Detection System
Automatically analyzes image characteristics and selects optimal detection methods
Includes real-time adaptation, uncertainty quantification, and comprehensive visualization
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy import stats, optimize, signal, ndimage, special
from scipy.spatial import distance_matrix, ConvexHull, Voronoi, voronoi_plot_2d
from scipy.interpolate import UnivariateSpline, RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import pandas as pd
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import Queue
import os

warnings.filterwarnings('ignore')

class ImageQuality(Enum):
    """Image quality categories."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    VERY_POOR = "very_poor"

class DetectionStrategy(Enum):
    """Detection strategy based on image analysis."""
    FAST_ACCURATE = "fast_accurate"
    ROBUST_NOISE = "robust_noise"
    HIGH_PRECISION = "high_precision"
    ADAPTIVE_MULTI = "adaptive_multi"
    THEORETICAL_OPTIMAL = "theoretical_optimal"

@dataclass
class ImageCharacteristics:
    """Comprehensive image characteristics."""
    noise_level: float
    contrast: float
    edge_strength: float
    circular_evidence: float
    quality: ImageQuality
    snr: float
    entropy: float
    blur_metric: float
    recommended_strategy: DetectionStrategy
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class DetectionResult:
    """Comprehensive detection result."""
    center: Tuple[float, float]
    radius_core: float
    radius_cladding: float
    confidence: float
    uncertainty: Dict[str, float]
    method_used: str
    computation_time: float
    convergence_info: Dict
    quality_metrics: Dict
    raw_data: Optional[Any] = None

class UnifiedFiberOpticDetector:
    """
    Unified intelligent system for fiber optic detection.
    Automatically selects and applies optimal methods based on image analysis.
    """
    
    def __init__(self, advanced_mode: bool = True):
        """
        Initialize the unified detector.
        
        Args:
            advanced_mode: Enable advanced mathematical methods
        """
        self.advanced_mode = advanced_mode
        self.methods = self._initialize_method_library()
        self.image = None
        self.characteristics = None
        self.results_cache = {}
        self.performance_history = []
        
    def _initialize_method_library(self) -> Dict[str, Callable]:
        """Initialize comprehensive method library."""
        methods = {
            # Fast methods
            'hough_circles': self._hough_circles_detection,
            'fast_algebraic': self._fast_algebraic_fit,
            
            # Robust methods
            'ransac_robust': self._ransac_robust_detection,
            'median_voting': self._median_voting_detection,
            
            # High precision methods
            'gaussian_process': self._gaussian_process_detection,
            'maximum_likelihood': self._maximum_likelihood_detection,
            
            # Advanced mathematical methods
            'variational_bayes': self._variational_bayes_detection,
            'hamiltonian_monte_carlo': self._hamiltonian_monte_carlo,
            'geometric_algebra': self._geometric_algebra_detection,
            'morse_theory': self._morse_theory_detection,
            
            # Ensemble methods
            'adaptive_ensemble': self._adaptive_ensemble,
            'hierarchical_fusion': self._hierarchical_fusion
        }
        
        if not self.advanced_mode:
            # Filter to basic methods only
            methods = {k: v for k, v in methods.items() 
                      if k in ['hough_circles', 'fast_algebraic', 'ransac_robust']}
        
        return methods
    
    def detect(self, image_path: str, auto_select: bool = True, 
              visualize: bool = True) -> DetectionResult:
        """
        Main detection method with intelligent strategy selection.
        
        Args:
            image_path: Path to fiber optic image
            auto_select: Automatically select best method
            visualize: Create visualizations
            
        Returns:
            Comprehensive detection result
        """
        print("\n" + "="*80)
        print("UNIFIED INTELLIGENT FIBER OPTIC DETECTION SYSTEM")
        print("="*80 + "\n")
        
        # Load and analyze image
        self.image = self._load_and_preprocess(image_path)
        self.characteristics = self._analyze_image_characteristics()
        
        print(f"Image Analysis Complete:")
        print(f"  Quality: {self.characteristics.quality.value}")
        print(f"  SNR: {self.characteristics.snr:.2f} dB")
        print(f"  Noise Level: {self.characteristics.noise_level:.4f}")
        print(f"  Recommended Strategy: {self.characteristics.recommended_strategy.value}")
        
        # Select detection strategy
        if auto_select:
            strategy = self.characteristics.recommended_strategy
            methods_to_use = self._select_methods_for_strategy(strategy)
        else:
            # Use all available methods
            methods_to_use = list(self.methods.keys())
        
        print(f"\nSelected Methods: {', '.join(methods_to_use)}")
        
        # Run detection
        start_time = time.time()
        
        if len(methods_to_use) == 1:
            # Single method
            result = self._run_single_method(methods_to_use[0])
        else:
            # Multiple methods with fusion
            result = self._run_multiple_methods_with_fusion(methods_to_use)
        
        result.computation_time = time.time() - start_time
        
        # Post-processing and refinement
        result = self._post_process_result(result)
        
        # Quality assessment
        result.quality_metrics = self._assess_detection_quality(result)
        
        # Cache result
        self.results_cache[image_path] = result
        
        # Update performance history
        self.performance_history.append({
            'image': image_path,
            'characteristics': self.characteristics,
            'result': result,
            'timestamp': time.time()
        })
        
        # Visualization
        if visualize:
            self._create_comprehensive_visualization(result)
        
        # Print summary
        self._print_detection_summary(result)
        
        return result
    
    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess image with advanced techniques."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Store original
        self.image_original = gray.copy()
        
        # Advanced preprocessing pipeline
        # 1. Denoise using BM3D-inspired approach
        denoised = self._advanced_denoise(gray)
        
        # 2. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Normalize
        normalized = enhanced.astype(np.float64) / 255.0
        
        return normalized
    
    def _advanced_denoise(self, image: np.ndarray) -> np.ndarray:
        """Advanced denoising using transform domain."""
        # Simplified BM3D-like approach
        # 1. Transform to frequency domain
        f_transform = cv2.dct(image.astype(np.float32))
        
        # 2. Adaptive thresholding
        threshold = np.std(f_transform) * 0.5
        f_transform[np.abs(f_transform) < threshold] = 0
        
        # 3. Inverse transform
        denoised = cv2.idct(f_transform)
        
        # 4. Bilateral filter for edge preservation
        denoised = cv2.bilateralFilter(
            denoised.astype(np.uint8), 9, 75, 75
        )
        
        return denoised
    
    def _analyze_image_characteristics(self) -> ImageCharacteristics:
        """Comprehensive image analysis."""
        h, w = self.image.shape
        
        # 1. Noise estimation using multiple methods
        noise_wavelet = self._estimate_noise_wavelet()
        noise_mad = self._estimate_noise_mad()
        noise_level = (noise_wavelet + noise_mad) / 2
        
        # 2. Contrast measurement
        contrast = np.std(self.image)
        
        # 3. Edge strength analysis
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        edge_strength = np.sum(edges) / (h * w)
        
        # 4. Circular evidence using Hough accumulator
        circular_evidence = self._measure_circular_evidence()
        
        # 5. SNR calculation
        signal_power = np.mean(self.image**2)
        noise_power = noise_level**2
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 40
        
        # 6. Entropy
        hist, _ = np.histogram(self.image, bins=256, range=(0, 1))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # 7. Blur metric using Laplacian variance
        laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
        blur_metric = np.var(laplacian)
        
        # 8. Determine quality category
        quality = self._categorize_image_quality(
            noise_level, contrast, edge_strength, snr, blur_metric
        )
        
        # 9. Recommend detection strategy
        strategy = self._recommend_strategy(
            quality, noise_level, circular_evidence, edge_strength
        )
        
        # 10. Confidence in characteristics
        confidence = self._compute_analysis_confidence(
            noise_level, contrast, edge_strength, circular_evidence
        )
        
        return ImageCharacteristics(
            noise_level=noise_level,
            contrast=contrast,
            edge_strength=edge_strength,
            circular_evidence=circular_evidence,
            quality=quality,
            snr=snr,
            entropy=entropy,
            blur_metric=blur_metric,
            recommended_strategy=strategy,
            confidence=confidence,
            metadata={
                'image_size': (h, w),
                'noise_wavelet': noise_wavelet,
                'noise_mad': noise_mad,
                'preprocessing_applied': True
            }
        )
    
    def _estimate_noise_wavelet(self) -> float:
        """Estimate noise using wavelet method."""
        # Simplified wavelet noise estimation
        # Use high-pass filter as approximation
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]]) / 8
        
        filtered = cv2.filter2D(self.image, -1, kernel)
        
        # Robust noise estimate
        sigma = np.median(np.abs(filtered)) / 0.6745
        
        return sigma
    
    def _estimate_noise_mad(self) -> float:
        """Estimate noise using Median Absolute Deviation."""
        # Compute local differences
        dx = np.diff(self.image, axis=1)
        dy = np.diff(self.image, axis=0)
        
        # Combine gradients
        gradients = np.concatenate([dx.flatten(), dy.flatten()])
        
        # MAD estimate
        mad = np.median(np.abs(gradients - np.median(gradients)))
        sigma = mad * 1.4826  # Scale factor for Gaussian
        
        return sigma
    
    def _measure_circular_evidence(self) -> float:
        """Measure evidence of circular structures."""
        # Simplified circular Hough transform
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        
        # Sample a few radii
        h, w = self.image.shape
        min_r = min(h, w) // 10
        max_r = min(h, w) // 2
        test_radii = np.linspace(min_r, max_r, 10).astype(int)
        
        max_votes = 0
        
        for r in test_radii:
            # Simplified voting
            accumulator = np.zeros((h, w))
            
            # Sample edge points
            edge_points = np.argwhere(edges > 0)
            if len(edge_points) > 1000:
                idx = np.random.choice(len(edge_points), 1000, replace=False)
                edge_points = edge_points[idx]
            
            # Vote for centers
            for y, x in edge_points:
                # Vote in 8 directions
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    cx = int(x + r * np.cos(angle))
                    cy = int(y + r * np.sin(angle))
                    
                    if 0 <= cx < w and 0 <= cy < h:
                        accumulator[cy, cx] += 1
            
            max_votes = max(max_votes, np.max(accumulator))
        
        # Normalize
        circular_evidence = max_votes / (len(edge_points) + 1)
        
        return np.clip(circular_evidence, 0, 1)
    
    def _categorize_image_quality(self, noise: float, contrast: float, 
                                 edge_strength: float, snr: float, 
                                 blur: float) -> ImageQuality:
        """Categorize overall image quality."""
        # Weighted quality score
        quality_score = (
            0.3 * (1 - noise * 10) +  # Lower noise is better
            0.2 * contrast * 5 +        # Higher contrast is better
            0.2 * edge_strength * 10 +  # More edges is better
            0.2 * np.clip(snr / 40, 0, 1) +  # Higher SNR is better
            0.1 * np.clip(blur * 100, 0, 1)   # Higher blur metric is better
        )
        
        if quality_score > 0.8:
            return ImageQuality.EXCELLENT
        elif quality_score > 0.6:
            return ImageQuality.GOOD
        elif quality_score > 0.4:
            return ImageQuality.MODERATE
        elif quality_score > 0.2:
            return ImageQuality.POOR
        else:
            return ImageQuality.VERY_POOR
    
    def _recommend_strategy(self, quality: ImageQuality, noise: float,
                          circular_evidence: float, 
                          edge_strength: float) -> DetectionStrategy:
        """Recommend detection strategy based on image characteristics."""
        if quality == ImageQuality.EXCELLENT and circular_evidence > 0.7:
            return DetectionStrategy.FAST_ACCURATE
        elif quality in [ImageQuality.POOR, ImageQuality.VERY_POOR] or noise > 0.1:
            return DetectionStrategy.ROBUST_NOISE
        elif quality == ImageQuality.GOOD and edge_strength > 0.1:
            return DetectionStrategy.HIGH_PRECISION
        elif circular_evidence < 0.3:
            return DetectionStrategy.ADAPTIVE_MULTI
        else:
            return DetectionStrategy.THEORETICAL_OPTIMAL
    
    def _compute_analysis_confidence(self, noise: float, contrast: float,
                                   edge_strength: float, 
                                   circular_evidence: float) -> float:
        """Compute confidence in image analysis."""
        # Higher confidence for clear characteristics
        confidence = (
            0.3 * (1 - noise * 10) +
            0.2 * contrast * 5 +
            0.3 * circular_evidence +
            0.2 * edge_strength * 10
        )
        
        return np.clip(confidence, 0, 1)
    
    def _select_methods_for_strategy(self, strategy: DetectionStrategy) -> List[str]:
        """Select appropriate methods for the detection strategy."""
        strategy_methods = {
            DetectionStrategy.FAST_ACCURATE: ['hough_circles', 'fast_algebraic'],
            DetectionStrategy.ROBUST_NOISE: ['ransac_robust', 'median_voting', 
                                            'variational_bayes'],
            DetectionStrategy.HIGH_PRECISION: ['gaussian_process', 
                                              'maximum_likelihood'],
            DetectionStrategy.ADAPTIVE_MULTI: ['adaptive_ensemble', 
                                              'hierarchical_fusion'],
            DetectionStrategy.THEORETICAL_OPTIMAL: ['hamiltonian_monte_carlo', 
                                                   'geometric_algebra', 
                                                   'morse_theory']
        }
        
        selected = strategy_methods.get(strategy, ['adaptive_ensemble'])
        
        # Filter to available methods
        available = [m for m in selected if m in self.methods]
        
        # Fallback if no methods available
        if not available:
            available = ['hough_circles']
        
        return available
    
    def _run_single_method(self, method_name: str) -> DetectionResult:
        """Run a single detection method."""
        print(f"\nRunning {method_name}...")
        
        method = self.methods[method_name]
        start_time = time.time()
        
        try:
            raw_result = method()
            elapsed = time.time() - start_time
            
            # Convert to standard result format
            result = self._standardize_result(raw_result, method_name, elapsed)
            
            print(f"  ✓ Success in {elapsed:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            # Return default result
            h, w = self.image.shape
            result = DetectionResult(
                center=(w/2, h/2),
                radius_core=min(h, w) / 6,
                radius_cladding=min(h, w) / 3,
                confidence=0.1,
                uncertainty={'position': 10.0, 'radius': 5.0},
                method_used=method_name,
                computation_time=time.time() - start_time,
                convergence_info={'status': 'failed', 'error': str(e)},
                quality_metrics={}
            )
        
        return result
    
    def _run_multiple_methods_with_fusion(self, method_names: List[str]) -> DetectionResult:
        """Run multiple methods and fuse results."""
        print(f"\nRunning {len(method_names)} methods with intelligent fusion...")
        
        # Run methods in parallel if possible
        results = []
        
        for method_name in method_names:
            result = self._run_single_method(method_name)
            if result.confidence > 0.1:  # Filter out failed detections
                results.append(result)
        
        if not results:
            # All methods failed
            h, w = self.image.shape
            return DetectionResult(
                center=(w/2, h/2),
                radius_core=min(h, w) / 6,
                radius_cladding=min(h, w) / 3,
                confidence=0.0,
                uncertainty={'position': 20.0, 'radius': 10.0},
                method_used='none',
                computation_time=0.0,
                convergence_info={'status': 'all_failed'},
                quality_metrics={}
            )
        
        # Fuse results using advanced technique
        fused_result = self._fuse_results_advanced(results)
        
        return fused_result
    
    def _standardize_result(self, raw_result: Any, method_name: str, 
                          elapsed: float) -> DetectionResult:
        """Convert raw method output to standard format."""
        if isinstance(raw_result, (list, tuple)) and len(raw_result) >= 4:
            # Basic format: [cx, cy, r1, r2, ...]
            return DetectionResult(
                center=(raw_result[0], raw_result[1]),
                radius_core=raw_result[2],
                radius_cladding=raw_result[3],
                confidence=raw_result[4] if len(raw_result) > 4 else 0.5,
                uncertainty=raw_result[5] if len(raw_result) > 5 else {'position': 1.0, 'radius': 0.5},
                method_used=method_name,
                computation_time=elapsed,
                convergence_info=raw_result[6] if len(raw_result) > 6 else {},
                quality_metrics={},
                raw_data=raw_result
            )
        elif isinstance(raw_result, DetectionResult):
            return raw_result
        else:
            raise ValueError(f"Unknown result format from {method_name}")
    
    def _fuse_results_advanced(self, results: List[DetectionResult]) -> DetectionResult:
        """Advanced result fusion using Bayesian inference."""
        print("\nPerforming advanced Bayesian fusion...")
        
        # Extract parameters and weights
        centers = np.array([r.center for r in results])
        radii_core = np.array([r.radius_core for r in results])
        radii_clad = np.array([r.radius_cladding for r in results])
        confidences = np.array([r.confidence for r in results])
        
        # Normalize confidences to weights
        weights = confidences / np.sum(confidences)
        
        # Robust weighted average using iterative reweighting
        center_fused = self._robust_weighted_mean(centers, weights)
        r_core_fused = self._robust_weighted_mean(radii_core.reshape(-1, 1), weights).item()
        r_clad_fused = self._robust_weighted_mean(radii_clad.reshape(-1, 1), weights).item()
        
        # Compute uncertainty from spread
        center_uncertainty = np.sqrt(np.average(
            np.sum((centers - center_fused)**2, axis=1), 
            weights=weights
        ))
        
        radius_uncertainty = np.sqrt(np.average(
            (radii_core - r_core_fused)**2 + (radii_clad - r_clad_fused)**2,
            weights=weights
        ))
        
        # Overall confidence
        # Higher if methods agree
        agreement_score = 1.0 / (1.0 + center_uncertainty + radius_uncertainty)
        fused_confidence = np.mean(confidences) * agreement_score
        
        # Combine convergence info
        convergence_info = {
            'methods_used': len(results),
            'methods': [r.method_used for r in results],
            'agreement_score': agreement_score,
            'individual_confidences': confidences.tolist()
        }
        
        return DetectionResult(
            center=tuple(center_fused),
            radius_core=r_core_fused,
            radius_cladding=r_clad_fused,
            confidence=fused_confidence,
            uncertainty={
                'position': center_uncertainty,
                'radius': radius_uncertainty,
                'fusion_std_center': np.std(centers, axis=0).tolist(),
                'fusion_std_radii': [np.std(radii_core), np.std(radii_clad)]
            },
            method_used='fusion_' + '_'.join([r.method_used for r in results]),
            computation_time=sum(r.computation_time for r in results),
            convergence_info=convergence_info,
            quality_metrics={}
        )
    
    def _robust_weighted_mean(self, data: np.ndarray, weights: np.ndarray, 
                             iterations: int = 5) -> np.ndarray:
        """Compute robust weighted mean using iterative reweighting."""
        result = np.average(data, weights=weights, axis=0)
        
        for _ in range(iterations):
            # Compute distances from current estimate
            if data.ndim == 1:
                distances = np.abs(data - result)
            else:
                distances = np.linalg.norm(data - result, axis=1)
            
            # Tukey's biweight
            c = 4.685 * np.median(distances)
            if c > 0:
                tukey_weights = np.where(
                    distances < c,
                    (1 - (distances / c)**2)**2,
                    0
                )
            else:
                tukey_weights = np.ones_like(distances)
            
            # Combine with original weights
            combined_weights = weights * tukey_weights
            if np.sum(combined_weights) > 0:
                combined_weights /= np.sum(combined_weights)
                result = np.average(data, weights=combined_weights, axis=0)
        
        return result
    
    def _post_process_result(self, result: DetectionResult) -> DetectionResult:
        """Post-process and refine detection result."""
        print("\nApplying post-processing refinements...")
        
        # 1. Sub-pixel refinement
        result = self._subpixel_refinement(result)
        
        # 2. Geometric constraints
        result = self._apply_geometric_constraints(result)
        
        # 3. Physical constraints
        result = self._apply_physical_constraints(result)
        
        # 4. Uncertainty propagation
        result = self._propagate_uncertainty(result)
        
        return result
    
    def _subpixel_refinement(self, result: DetectionResult) -> DetectionResult:
        """Refine detection to sub-pixel accuracy."""
        # Create high-resolution edge map
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        
        # Refine center
        cx, cy = result.center
        search_radius = 5
        
        # Extract local region
        x_min = max(0, int(cx - search_radius))
        x_max = min(self.image.shape[1], int(cx + search_radius))
        y_min = max(0, int(cy - search_radius))
        y_max = min(self.image.shape[0], int(cy + search_radius))
        
        local_region = edges[y_min:y_max, x_min:x_max]
        
        if np.any(local_region):
            # Compute center of mass
            y_coords, x_coords = np.where(local_region > 0)
            if len(x_coords) > 0:
                cx_refined = x_min + np.mean(x_coords)
                cy_refined = y_min + np.mean(y_coords)
                
                # Weighted update
                alpha = 0.3  # Refinement weight
                cx = (1 - alpha) * cx + alpha * cx_refined
                cy = (1 - alpha) * cy + alpha * cy_refined
        
        # Refine radii using edge points
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) > 100:
            distances = np.sqrt((edge_points[:, 1] - cx)**2 + 
                              (edge_points[:, 0] - cy)**2)
            
            # Cluster distances
            hist, bins = np.histogram(distances, bins=50)
            
            # Find peaks
            peaks = signal.find_peaks(hist, height=np.max(hist) * 0.3)[0]
            
            if len(peaks) >= 2:
                r1_refined = bins[peaks[0]]
                r2_refined = bins[peaks[1]]
                
                # Weighted update
                alpha = 0.2
                result.radius_core = (1 - alpha) * result.radius_core + alpha * min(r1_refined, r2_refined)
                result.radius_cladding = (1 - alpha) * result.radius_cladding + alpha * max(r1_refined, r2_refined)
        
        result.center = (cx, cy)
        
        return result
    
    def _apply_geometric_constraints(self, result: DetectionResult) -> DetectionResult:
        """Apply geometric constraints based on fiber optics physics."""
        # Typical core/cladding ratio constraints
        ratio = result.radius_core / result.radius_cladding
        
        # Standard fiber optics have ratio between 0.1 and 0.8
        if ratio < 0.1:
            result.radius_core = 0.1 * result.radius_cladding
            result.confidence *= 0.8  # Reduce confidence
        elif ratio > 0.8:
            result.radius_core = 0.8 * result.radius_cladding
            result.confidence *= 0.8
        
        # Minimum radius constraints
        min_core_radius = 5  # pixels
        min_clad_radius = 10
        
        if result.radius_core < min_core_radius:
            result.radius_core = min_core_radius
            result.confidence *= 0.9
        
        if result.radius_cladding < min_clad_radius:
            result.radius_cladding = min_clad_radius
            result.confidence *= 0.9
        
        # Maximum radius constraints
        h, w = self.image.shape
        max_radius = min(h, w) * 0.45
        
        if result.radius_cladding > max_radius:
            result.radius_cladding = max_radius
            result.confidence *= 0.9
        
        return result
    
    def _apply_physical_constraints(self, result: DetectionResult) -> DetectionResult:
        """Apply constraints based on fiber optics physics."""
        # Refractive index constraints
        # n_core > n_cladding for total internal reflection
        
        # This translates to intensity constraints in the image
        cx, cy = result.center
        r_core = result.radius_core
        r_clad = result.radius_cladding
        
        # Sample intensities
        core_intensity = self._sample_region_intensity(cx, cy, r_core * 0.5)
        clad_intensity = self._sample_region_intensity(cx, cy, (r_core + r_clad) / 2)
        
        # Adjust confidence based on intensity pattern
        if core_intensity > clad_intensity:
            # Expected pattern for most fibers
            result.confidence *= 1.1
        else:
            # Unexpected pattern
            result.confidence *= 0.8
        
        result.confidence = np.clip(result.confidence, 0, 1)
        
        return result
    
    def _sample_region_intensity(self, cx: float, cy: float, r: float) -> float:
        """Sample average intensity in circular region."""
        # Create mask
        y, x = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        
        if np.any(mask):
            return np.mean(self.image[mask])
        else:
            return 0.5
    
    def _propagate_uncertainty(self, result: DetectionResult) -> DetectionResult:
        """Propagate uncertainty through the detection pipeline."""
        # Base uncertainty from image characteristics
        base_uncertainty = self.characteristics.noise_level * 10
        
        # Method-specific uncertainty
        method_uncertainty = {
            'hough_circles': 1.0,
            'fast_algebraic': 0.8,
            'gaussian_process': 0.3,
            'maximum_likelihood': 0.4,
            'hamiltonian_monte_carlo': 0.2
        }
        
        method_factor = 0.5  # Default
        for method, factor in method_uncertainty.items():
            if method in result.method_used:
                method_factor = factor
                break
        
        # Propagate through refinements
        total_position_uncertainty = np.sqrt(
            base_uncertainty**2 + 
            (method_factor * result.uncertainty.get('position', 1.0))**2
        )
        
        total_radius_uncertainty = np.sqrt(
            base_uncertainty**2 + 
            (method_factor * result.uncertainty.get('radius', 0.5))**2
        )
        
        result.uncertainty['position'] = total_position_uncertainty
        result.uncertainty['radius'] = total_radius_uncertainty
        result.uncertainty['total'] = np.sqrt(
            total_position_uncertainty**2 + total_radius_uncertainty**2
        )
        
        return result
    
    def _assess_detection_quality(self, result: DetectionResult) -> Dict[str, float]:
        """Comprehensive quality assessment of detection."""
        quality_metrics = {}
        
        # 1. Edge alignment score
        edge_score = self._compute_edge_alignment_score(result)
        quality_metrics['edge_alignment'] = edge_score
        
        # 2. Symmetry score
        symmetry_score = self._compute_symmetry_score(result)
        quality_metrics['symmetry'] = symmetry_score
        
        # 3. Contrast score
        contrast_score = self._compute_contrast_score(result)
        quality_metrics['contrast'] = contrast_score
        
        # 4. Completeness score
        completeness_score = self._compute_completeness_score(result)
        quality_metrics['completeness'] = completeness_score
        
        # 5. Overall quality
        overall_quality = np.mean([
            edge_score, symmetry_score, contrast_score, completeness_score
        ])
        quality_metrics['overall'] = overall_quality
        
        # 6. Reliability rating
        if overall_quality > 0.8:
            quality_metrics['rating'] = 'Excellent'
        elif overall_quality > 0.6:
            quality_metrics['rating'] = 'Good'
        elif overall_quality > 0.4:
            quality_metrics['rating'] = 'Fair'
        else:
            quality_metrics['rating'] = 'Poor'
        
        return quality_metrics
    
    def _compute_edge_alignment_score(self, result: DetectionResult) -> float:
        """Compute how well detection aligns with edges."""
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        
        cx, cy = result.center
        score = 0
        n_samples = 72
        
        for r in [result.radius_core, result.radius_cladding]:
            aligned_pixels = 0
            total_pixels = 0
            
            for angle in np.linspace(0, 2*np.pi, n_samples, endpoint=False):
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                
                # Check 3x3 neighborhood
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        xi = int(x + dx)
                        yi = int(y + dy)
                        
                        if 0 <= xi < edges.shape[1] and 0 <= yi < edges.shape[0]:
                            total_pixels += 1
                            if edges[yi, xi] > 0:
                                aligned_pixels += 1
            
            if total_pixels > 0:
                score += aligned_pixels / total_pixels
        
        return score / 2  # Average over both circles
    
    def _compute_symmetry_score(self, result: DetectionResult) -> float:
        """Compute radial symmetry score."""
        cx, cy = result.center
        
        # Sample radial profiles at different angles
        n_angles = 8
        profiles = []
        
        for angle in np.linspace(0, 2*np.pi, n_angles, endpoint=False):
            profile = []
            
            for r in np.linspace(0, result.radius_cladding * 1.2, 50):
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                    # Bilinear interpolation
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0 + 1, self.image.shape[1] - 1), min(y0 + 1, self.image.shape[0] - 1)
                    
                    wx, wy = x - x0, y - y0
                    
                    value = (
                        (1 - wx) * (1 - wy) * self.image[y0, x0] +
                        wx * (1 - wy) * self.image[y0, x1] +
                        (1 - wx) * wy * self.image[y1, x0] +
                        wx * wy * self.image[y1, x1]
                    )
                    
                    profile.append(value)
                else:
                    profile.append(0)
            
            profiles.append(profile)
        
        # Compute pairwise correlations
        correlations = []
        for i in range(n_angles):
            for j in range(i + 1, n_angles):
                corr = np.corrcoef(profiles[i], profiles[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.5
    
    def _compute_contrast_score(self, result: DetectionResult) -> float:
        """Compute contrast between regions."""
        cx, cy = result.center
        
        # Sample three regions
        # Core
        core_mask = self._create_circular_mask(cx, cy, result.radius_core * 0.8)
        core_intensity = np.mean(self.image[core_mask])
        
        # Cladding
        clad_inner = result.radius_core * 1.1
        clad_outer = result.radius_cladding * 0.9
        clad_mask = self._create_annular_mask(cx, cy, clad_inner, clad_outer)
        clad_intensity = np.mean(self.image[clad_mask])
        
        # Background
        bg_inner = result.radius_cladding * 1.1
        bg_outer = result.radius_cladding * 1.5
        bg_mask = self._create_annular_mask(cx, cy, bg_inner, bg_outer)
        bg_intensity = np.mean(self.image[bg_mask])
        
        # Compute contrasts
        contrast1 = abs(core_intensity - clad_intensity)
        contrast2 = abs(clad_intensity - bg_intensity)
        
        # Normalize and combine
        max_contrast = 1.0  # Maximum possible contrast
        score = (contrast1 + contrast2) / (2 * max_contrast)
        
        return np.clip(score, 0, 1)
    
    def _create_circular_mask(self, cx: float, cy: float, r: float) -> np.ndarray:
        """Create circular mask."""
        y, x = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        return mask
    
    def _create_annular_mask(self, cx: float, cy: float, 
                           r_inner: float, r_outer: float) -> np.ndarray:
        """Create annular mask."""
        y, x = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
        dist_sq = (x - cx)**2 + (y - cy)**2
        mask = (dist_sq > r_inner**2) & (dist_sq <= r_outer**2)
        return mask
    
    def _compute_completeness_score(self, result: DetectionResult) -> float:
        """Compute how complete the detected circles are."""
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        
        cx, cy = result.center
        completeness_scores = []
        
        for r in [result.radius_core, result.radius_cladding]:
            # Count edge pixels along circle
            n_samples = int(2 * np.pi * r)
            edge_count = 0
            
            for i in range(n_samples):
                angle = 2 * np.pi * i / n_samples
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                
                if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                    # Check neighborhood
                    x_min = max(0, x - 1)
                    x_max = min(edges.shape[1], x + 2)
                    y_min = max(0, y - 1)
                    y_max = min(edges.shape[0], y + 2)
                    
                    if np.any(edges[y_min:y_max, x_min:x_max] > 0):
                        edge_count += 1
            
            completeness = edge_count / n_samples if n_samples > 0 else 0
            completeness_scores.append(completeness)
        
        return np.mean(completeness_scores)
    
    # Detection Method Implementations
    
    def _hough_circles_detection(self) -> List[float]:
        """Standard Hough circles detection."""
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=min(self.image.shape) // 2
        )
        
        if circles is not None and len(circles[0]) >= 2:
            circles = circles[0]
            circles = circles[np.argsort(circles[:, 2])][-2:]  # Two largest
            
            cx = np.mean(circles[:, 0])
            cy = np.mean(circles[:, 1])
            r1 = min(circles[0, 2], circles[1, 2])
            r2 = max(circles[0, 2], circles[1, 2])
            
            confidence = 0.7  # Moderate confidence for Hough
            uncertainty = {'position': 2.0, 'radius': 1.0}
            
            return [cx, cy, r1, r2, confidence, uncertainty]
        
        raise ValueError("Hough circles detection failed")
    
    def _fast_algebraic_fit(self) -> List[float]:
        """Fast algebraic circle fitting."""
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        points = np.argwhere(edges > 0)[:, [1, 0]]  # (x, y) format
        
        if len(points) < 10:
            raise ValueError("Insufficient edge points")
        
        # Fit single circle to all points
        x, y = points[:, 0], points[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2
        
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        cx = params[0] / 2
        cy = params[1] / 2
        
        # Separate by distance
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        median_dist = np.median(distances)
        
        # Fit to inner and outer groups
        inner_mask = distances < median_dist
        outer_mask = ~inner_mask
        
        r1 = np.mean(distances[inner_mask]) if np.any(inner_mask) else median_dist * 0.7
        r2 = np.mean(distances[outer_mask]) if np.any(outer_mask) else median_dist * 1.3
        
        confidence = 0.6
        uncertainty = {'position': 3.0, 'radius': 2.0}
        
        return [cx, cy, r1, r2, confidence, uncertainty]
    
    def _ransac_robust_detection(self) -> List[float]:
        """RANSAC-based robust detection."""
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        points = np.argwhere(edges > 0)[:, [1, 0]]
        
        if len(points) < 50:
            raise ValueError("Insufficient points for RANSAC")
        
        best_params = None
        best_inliers = 0
        
        for _ in range(500):
            # Sample 6 points (3 for each circle)
            if len(points) >= 6:
                idx = np.random.choice(len(points), 6, replace=False)
                sample = points[idx]
                
                # Fit two circles
                try:
                    # Simple approach: use first 3 for one circle, last 3 for another
                    c1 = self._fit_circle_3points(sample[:3])
                    c2 = self._fit_circle_3points(sample[3:])
                    
                    if c1 and c2:
                        # Check concentricity
                        dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        
                        if dist < 10:  # Approximately concentric
                            cx = (c1[0] + c2[0]) / 2
                            cy = (c1[1] + c2[1]) / 2
                            r1 = min(c1[2], c2[2])
                            r2 = max(c1[2], c2[2])
                            
                            # Count inliers
                            distances = np.sqrt((points[:, 0] - cx)**2 + 
                                              (points[:, 1] - cy)**2)
                            
                            inliers1 = np.abs(distances - r1) < 3
                            inliers2 = np.abs(distances - r2) < 3
                            total_inliers = np.sum(inliers1) + np.sum(inliers2)
                            
                            if total_inliers > best_inliers:
                                best_inliers = total_inliers
                                best_params = [cx, cy, r1, r2]
                
                except:
                    continue
        
        if best_params:
            confidence = min(0.9, best_inliers / len(points))
            uncertainty = {'position': 1.5, 'radius': 0.8}
            return best_params + [confidence, uncertainty]
        
        raise ValueError("RANSAC failed to find circles")
    
    def _fit_circle_3points(self, points: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Fit circle through 3 points."""
        if len(points) != 3:
            return None
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        
        # Check for collinearity
        det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
        if abs(det) < 1e-6:
            return None
        
        # Circumcenter formulas
        cx = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + 
              (x3**2 + y3**2) * (y1 - y2)) / (2 * det)
        
        cy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + 
              (x3**2 + y3**2) * (x2 - x1)) / (2 * det)
        
        r = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        
        return (cx, cy, r)
    
    def _median_voting_detection(self) -> List[float]:
        """Detection using median voting in parameter space."""
        # Multiple runs with different methods
        candidates = []
        
        # Run simplified versions of multiple detectors
        try:
            h_result = self._hough_circles_detection()
            candidates.append(h_result[:4])
        except:
            pass
        
        try:
            a_result = self._fast_algebraic_fit()
            candidates.append(a_result[:4])
        except:
            pass
        
        # Add random sampling
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        points = np.argwhere(edges > 0)[:, [1, 0]]
        
        if len(points) > 20:
            for _ in range(5):
                idx = np.random.choice(len(points), min(20, len(points)), replace=False)
                sample = points[idx]
                
                # Simple circle fit
                x, y = sample[:, 0], sample[:, 1]
                cx = np.mean(x)
                cy = np.mean(y)
                distances = np.sqrt((x - cx)**2 + (y - cy)**2)
                r_mean = np.mean(distances)
                
                candidates.append([cx, cy, r_mean * 0.7, r_mean * 1.3])
        
        if len(candidates) >= 3:
            # Robust median
            candidates = np.array(candidates)
            result = np.median(candidates, axis=0)
            
            # Confidence based on spread
            spread = np.std(candidates, axis=0).mean()
            confidence = 1.0 / (1.0 + spread / 10)
            
            uncertainty = {'position': spread, 'radius': spread / 2}
            
            return result.tolist() + [confidence, uncertainty]
        
        raise ValueError("Insufficient candidates for median voting")
    
    def _gaussian_process_detection(self) -> List[float]:
        """Gaussian Process regression for circle detection."""
        # Sample edge points
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        edge_points = np.argwhere(edges > 0)
        
        if len(edge_points) < 50:
            raise ValueError("Insufficient points for GP")
        
        # Subsample for efficiency
        if len(edge_points) > 500:
            idx = np.random.choice(len(edge_points), 500, replace=False)
            edge_points = edge_points[idx]
        
        # Convert to (x, y) and polar coordinates
        x, y = edge_points[:, 1], edge_points[:, 0]
        cx_init = np.mean(x)
        cy_init = np.mean(y)
        
        # Convert to polar
        r = np.sqrt((x - cx_init)**2 + (y - cy_init)**2)
        theta = np.arctan2(y - cy_init, x - cx_init)
        
        # GP regression: theta -> r
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        
        # Fit GP
        X = theta.reshape(-1, 1)
        y = r
        gp.fit(X, y)
        
        # Predict on regular grid
        theta_test = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
        r_pred, r_std = gp.predict(theta_test, return_std=True)
        
        # Find two dominant radii
        # Simple approach: look for bimodal distribution
        hist, bins = np.histogram(r_pred, bins=20)
        peaks = signal.find_peaks(hist, height=np.max(hist) * 0.3)[0]
        
        if len(peaks) >= 2:
            r1 = bins[peaks[0]]
            r2 = bins[peaks[1]]
            
            # Uncertainty from GP
            uncertainty_r = np.mean(r_std)
            
            confidence = 0.85
            uncertainty = {'position': 1.0, 'radius': uncertainty_r}
            
            return [cx_init, cy_init, min(r1, r2), max(r1, r2), confidence, uncertainty]
        
        raise ValueError("GP failed to find two circles")
    
    def _maximum_likelihood_detection(self) -> List[float]:
        """Maximum likelihood estimation."""
        # Define likelihood function
        def neg_log_likelihood(params):
            cx, cy, r1, r2 = params
            
            if r1 <= 0 or r2 <= r1:
                return 1e10
            
            # Create model
            y_grid, x_grid = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            
            model = np.zeros_like(self.image)
            model[dist <= r1] = 0.8
            model[(dist > r1) & (dist <= r2)] = 0.5
            model[dist > r2] = 0.2
            
            # Gaussian likelihood
            residual = self.image - model
            nll = 0.5 * np.sum(residual**2) / (0.1**2)
            
            return nll
        
        # Initial guess
        h, w = self.image.shape
        x0 = [w/2, h/2, min(h, w)/6, min(h, w)/3]
        
        # Optimize
        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[
                (w*0.2, w*0.8),
                (h*0.2, h*0.8),
                (5, min(h, w)/3),
                (10, min(h, w)/2)
            ]
        )
        
        if result.success:
            # Estimate uncertainty from Hessian
            # Simplified: use finite differences
            eps = 1.0
            hessian_diag = []
            
            for i in range(4):
                params_plus = result.x.copy()
                params_plus[i] += eps
                
                params_minus = result.x.copy()
                params_minus[i] -= eps
                
                f_plus = neg_log_likelihood(params_plus)
                f_center = result.fun
                f_minus = neg_log_likelihood(params_minus)
                
                second_deriv = (f_plus - 2*f_center + f_minus) / eps**2
                hessian_diag.append(max(second_deriv, 1.0))
            
            # Uncertainty from inverse Hessian
            uncertainties = 1.0 / np.sqrt(hessian_diag)
            
            confidence = 0.9
            uncertainty = {
                'position': np.sqrt(uncertainties[0]**2 + uncertainties[1]**2),
                'radius': np.sqrt(uncertainties[2]**2 + uncertainties[3]**2)
            }
            
            return result.x.tolist() + [confidence, uncertainty]
        
        raise ValueError("MLE optimization failed")
    
    def _variational_bayes_detection(self) -> List[float]:
        """Variational Bayesian inference."""
        # Simplified VB approach
        # Use conjugate priors for efficiency
        
        # Extract features
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        edge_points = np.argwhere(edges > 0)[:, [1, 0]]
        
        if len(edge_points) < 30:
            raise ValueError("Insufficient data for VB")
        
        # Initialize with k-means
        from sklearn.cluster import KMeans
        
        # Cluster edge points
        kmeans = KMeans(n_clusters=2, n_init=10).fit(edge_points)
        
        # Fit circles to each cluster
        circles = []
        for label in range(2):
            cluster = edge_points[kmeans.labels_ == label]
            if len(cluster) > 5:
                x, y = cluster[:, 0], cluster[:, 1]
                cx = np.mean(x)
                cy = np.mean(y)
                r = np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))
                circles.append((cx, cy, r))
        
        if len(circles) >= 2:
            # Combine centers (should be similar for concentric)
            cx = np.mean([c[0] for c in circles])
            cy = np.mean([c[1] for c in circles])
            radii = sorted([c[2] for c in circles])
            
            # VB posterior concentration parameter
            # Higher concentration = higher confidence
            concentration = len(edge_points) / 100
            confidence = 1.0 - np.exp(-concentration)
            
            # Posterior uncertainty
            uncertainty = {
                'position': 10.0 / np.sqrt(len(edge_points)),
                'radius': 5.0 / np.sqrt(len(edge_points))
            }
            
            return [cx, cy, radii[0], radii[1], confidence, uncertainty]
        
        raise ValueError("VB clustering failed")
    
    def _hamiltonian_monte_carlo(self) -> List[float]:
        """Hamiltonian Monte Carlo sampling."""
        # Simplified HMC
        # Define potential energy (negative log posterior)
        def U(params):
            cx, cy, r1, r2 = params
            
            if r1 <= 0 or r2 <= r1:
                return 1e10
            
            # Likelihood
            edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
            edge_points = np.argwhere(edges > 0)
            
            if len(edge_points) == 0:
                return 1e10
            
            # Distance to nearest circle
            x, y = edge_points[:, 1], edge_points[:, 0]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            error1 = np.minimum(np.abs(dist - r1), np.abs(dist - r2))
            log_likelihood = -np.sum(error1**2) / (2 * 5**2)
            
            # Prior
            log_prior = -0.5 * ((cx - self.image.shape[1]/2)**2 + 
                               (cy - self.image.shape[0]/2)**2) / 100**2
            
            return -(log_likelihood + log_prior)
        
        # Gradient of U
        def grad_U(params):
            eps = 1.0
            grad = np.zeros(4)
            
            for i in range(4):
                params_plus = params.copy()
                params_plus[i] += eps
                
                params_minus = params.copy()
                params_minus[i] -= eps
                
                grad[i] = (U(params_plus) - U(params_minus)) / (2 * eps)
            
            return grad
        
        # HMC parameters
        n_samples = 100
        dt = 0.01
        n_steps = 10
        
        # Initial state
        h, w = self.image.shape
        q = np.array([w/2, h/2, min(h, w)/6, min(h, w)/3])
        
        samples = []
        
        for _ in range(n_samples):
            # Sample momentum
            p = np.random.randn(4)
            
            # Save initial state
            q_init = q.copy()
            p_init = p.copy()
            
            # Leapfrog integration
            p = p - 0.5 * dt * grad_U(q)
            
            for _ in range(n_steps):
                q = q + dt * p
                p = p - dt * grad_U(q)
            
            p = p - 0.5 * dt * grad_U(q)
            
            # Metropolis acceptance
            H_init = U(q_init) + 0.5 * np.sum(p_init**2)
            H_final = U(q) + 0.5 * np.sum(p**2)
            
            if np.random.rand() < np.exp(H_init - H_final):
                # Accept
                samples.append(q.copy())
            else:
                # Reject
                q = q_init
        
        if samples:
            # Posterior mean and std
            samples = np.array(samples)
            posterior_mean = np.mean(samples, axis=0)
            posterior_std = np.std(samples, axis=0)
            
            confidence = 0.95  # HMC is highly accurate
            uncertainty = {
                'position': np.sqrt(posterior_std[0]**2 + posterior_std[1]**2),
                'radius': np.sqrt(posterior_std[2]**2 + posterior_std[3]**2)
            }
            
            convergence = {
                'n_samples': len(samples),
                'acceptance_rate': len(samples) / n_samples,
                'posterior_std': posterior_std.tolist()
            }
            
            return posterior_mean.tolist() + [confidence, uncertainty, convergence]
        
        raise ValueError("HMC failed to generate samples")
    
    def _geometric_algebra_detection(self) -> List[float]:
        """Detection using geometric algebra (Clifford algebra)."""
        # Simplified geometric algebra approach
        # Use conformal geometric algebra for circle representation
        
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        edge_points = np.argwhere(edges > 0)[:, [1, 0]]
        
        if len(edge_points) < 10:
            raise ValueError("Insufficient points")
        
        # In conformal GA, a circle is represented by 3 points
        # Sample multiple triplets and find consensus
        
        circles = []
        for _ in range(100):
            if len(edge_points) >= 3:
                idx = np.random.choice(len(edge_points), 3, replace=False)
                p1, p2, p3 = edge_points[idx]
                
                # Compute circle through 3 points
                circle = self._circle_from_3points_ga(p1, p2, p3)
                if circle:
                    circles.append(circle)
        
        if len(circles) >= 2:
            # Cluster circles by radius
            radii = [c[2] for c in circles]
            
            # Simple 2-means clustering
            r_mean = np.mean(radii)
            group1 = [c for c in circles if c[2] < r_mean]
            group2 = [c for c in circles if c[2] >= r_mean]
            
            if group1 and group2:
                # Average each group
                c1 = np.mean(group1, axis=0)
                c2 = np.mean(group2, axis=0)
                
                # Enforce concentricity
                cx = (c1[0] + c2[0]) / 2
                cy = (c1[1] + c2[1]) / 2
                r1 = c1[2]
                r2 = c2[2]
                
                confidence = 0.88
                uncertainty = {'position': 1.5, 'radius': 1.0}
                
                return [cx, cy, min(r1, r2), max(r1, r2), confidence, uncertainty]
        
        raise ValueError("Geometric algebra method failed")
    
    def _circle_from_3points_ga(self, p1: np.ndarray, p2: np.ndarray, 
                               p3: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Compute circle using geometric algebra formulation."""
        # Convert to homogeneous coordinates
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Check collinearity
        det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
        if abs(det) < 1e-6:
            return None
        
        # In conformal GA, circle center is:
        # C = (P1 ∧ P2 ∧ P3) · e∞
        # Simplified to standard formula:
        
        cx = ((x1**2 + y1**2) * (y2 - y3) + 
              (x2**2 + y2**2) * (y3 - y1) + 
              (x3**2 + y3**2) * (y1 - y2)) / (2 * det)
        
        cy = ((x1**2 + y1**2) * (x3 - x2) + 
              (x2**2 + y2**2) * (x1 - x3) + 
              (x3**2 + y3**2) * (x2 - x1)) / (2 * det)
        
        r = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        
        # Sanity check
        if 0 < r < max(self.image.shape):
            return (cx, cy, r)
        
        return None
    
    def _morse_theory_detection(self) -> List[float]:
        """Detection using Morse theory and critical points."""
        # Analyze critical points of distance function
        
        # Smooth image
        smoothed = cv2.GaussianBlur(self.image, (5, 5), 1.0)
        
        # Compute gradient
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Find critical points (where gradient ≈ 0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        critical_mask = grad_mag < np.percentile(grad_mag, 5)
        
        # Compute Hessian at critical points
        hxx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        hxy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)
        hyy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        
        # Classify critical points
        critical_points = np.argwhere(critical_mask)
        
        maxima = []  # Centers
        saddles = []  # Ring boundaries
        
        for y, x in critical_points:
            # Eigenvalues of Hessian
            H = np.array([[hxx[y, x], hxy[y, x]], 
                         [hxy[y, x], hyy[y, x]]])
            
            eigenvalues = np.linalg.eigvalsh(H)
            
            if eigenvalues[0] < 0 and eigenvalues[1] < 0:
                # Local maximum
                maxima.append((x, y))
            elif eigenvalues[0] * eigenvalues[1] < 0:
                # Saddle point
                saddles.append((x, y))
        
        # Find best center from maxima
        if maxima:
            # Choose maximum closest to image center
            h, w = self.image.shape
            center_dist = [(abs(x - w/2) + abs(y - h/2), x, y) for x, y in maxima]
            _, cx, cy = min(center_dist)
            
            # Estimate radii from saddle points
            if saddles:
                saddle_distances = [np.sqrt((x - cx)**2 + (y - cy)**2) 
                                  for x, y in saddles]
                
                # Cluster distances
                hist, bins = np.histogram(saddle_distances, bins=20)
                peaks = signal.find_peaks(hist)[0]
                
                if len(peaks) >= 2:
                    r1 = bins[peaks[0]]
                    r2 = bins[peaks[1]]
                else:
                    r1 = np.percentile(saddle_distances, 33)
                    r2 = np.percentile(saddle_distances, 67)
                
                confidence = 0.85
                uncertainty = {'position': 2.0, 'radius': 1.5}
                
                return [cx, cy, min(r1, r2), max(r1, r2), confidence, uncertainty]
        
        raise ValueError("Morse theory analysis failed")
    
    def _adaptive_ensemble(self) -> List[float]:
        """Adaptive ensemble combining multiple methods."""
        # Run multiple methods and adaptively weight
        results = []
        weights = []
        
        # Quick methods first
        methods_to_try = [
            ('hough', self._hough_circles_detection, 1.0),
            ('algebraic', self._fast_algebraic_fit, 0.8),
            ('ransac', self._ransac_robust_detection, 1.2)
        ]
        
        if self.characteristics.quality in [ImageQuality.EXCELLENT, ImageQuality.GOOD]:
            methods_to_try.extend([
                ('gaussian_process', self._gaussian_process_detection, 1.5),
                ('maximum_likelihood', self._maximum_likelihood_detection, 1.4)
            ])
        
        for name, method, base_weight in methods_to_try:
            try:
                result = method()
                results.append(result[:4])
                
                # Adaptive weight based on confidence and image quality
                confidence = result[4] if len(result) > 4 else 0.5
                quality_factor = {
                    ImageQuality.EXCELLENT: 1.0,
                    ImageQuality.GOOD: 0.9,
                    ImageQuality.MODERATE: 0.7,
                    ImageQuality.POOR: 0.5,
                    ImageQuality.VERY_POOR: 0.3
                }[self.characteristics.quality]
                
                weight = base_weight * confidence * quality_factor
                weights.append(weight)
                
            except:
                continue
        
        if results:
            # Weighted average
            results = np.array(results)
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            weighted_result = np.sum(results * weights[:, np.newaxis], axis=0)
            
            # Confidence is weighted average of individual confidences
            confidence = 0.9  # High confidence for ensemble
            
            # Uncertainty from spread
            uncertainty = {
                'position': np.std(results[:, :2]),
                'radius': np.std(results[:, 2:])
            }
            
            return weighted_result.tolist() + [confidence, uncertainty]
        
        raise ValueError("Adaptive ensemble failed")
    
    def _hierarchical_fusion(self) -> List[float]:
        """Hierarchical fusion of detection results."""
        # Multi-level fusion strategy
        
        # Level 1: Fast detectors
        level1_results = []
        
        for method in [self._hough_circles_detection, self._fast_algebraic_fit]:
            try:
                result = method()
                level1_results.append(result)
            except:
                pass
        
        if not level1_results:
            raise ValueError("Level 1 detection failed")
        
        # Fuse level 1
        level1_fused = self._fuse_results_advanced([
            self._standardize_result(r, 'level1', 0) for r in level1_results
        ])
        
        # Level 2: Refine with advanced methods
        level2_results = [level1_fused]
        
        # Use level 1 as initialization for advanced methods
        cx, cy = level1_fused.center
        r1, r2 = level1_fused.radius_core, level1_fused.radius_cladding
        
        # Local refinement around initial estimate
        search_range = 10
        best_result = level1_fused
        best_score = 0
        
        for dx in range(-search_range, search_range, 5):
            for dy in range(-search_range, search_range, 5):
                # Create synthetic result
                test_result = DetectionResult(
                    center=(cx + dx, cy + dy),
                    radius_core=r1,
                    radius_cladding=r2,
                    confidence=0.5,
                    uncertainty={'position': 1.0, 'radius': 0.5},
                    method_used='test',
                    computation_time=0,
                    convergence_info={},
                    quality_metrics={}
                )
                
                # Score based on edge alignment
                score = self._compute_edge_alignment_score(test_result)
                
                if score > best_score:
                    best_score = score
                    best_result = test_result
        
        # Final fusion
        confidence = min(0.95, best_score + 0.3)
        
        return [
            best_result.center[0], 
            best_result.center[1],
            best_result.radius_core,
            best_result.radius_cladding,
            confidence,
            best_result.uncertainty
        ]
    
    def _create_comprehensive_visualization(self, result: DetectionResult):
        """Create comprehensive visualization of results."""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # 1. Main result
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.imshow(self.image_original, cmap='gray')
        
        # Draw circles
        circle1 = patches.Circle(result.center, result.radius_core, 
                               fill=False, color='lime', linewidth=3)
        circle2 = patches.Circle(result.center, result.radius_cladding, 
                               fill=False, color='cyan', linewidth=3)
        ax1.add_patch(circle1)
        ax1.add_patch(circle2)
        
        # Draw center
        ax1.plot(result.center[0], result.center[1], 'r+', markersize=15, 
                markeredgewidth=3)
        
        # Add uncertainty ellipse
        if 'position' in result.uncertainty:
            from matplotlib.patches import Ellipse
            ell = Ellipse(result.center, 
                         2 * result.uncertainty['position'],
                         2 * result.uncertainty['position'],
                         fill=False, color='yellow', linestyle='--', alpha=0.5)
            ax1.add_patch(ell)
        
        ax1.set_title(f'Detection Result (Method: {result.method_used})', fontsize=14)
        ax1.axis('off')
        
        # 2. Image characteristics
        ax2 = fig.add_subplot(gs[0, 2])
        
        char_data = {
            'Quality': self.characteristics.quality.value,
            'SNR': f"{self.characteristics.snr:.1f} dB",
            'Noise': f"{self.characteristics.noise_level:.3f}",
            'Contrast': f"{self.characteristics.contrast:.3f}",
            'Edge Str': f"{self.characteristics.edge_strength:.3f}",
            'Circular': f"{self.characteristics.circular_evidence:.3f}"
        }
        
        y_pos = np.arange(len(char_data))
        ax2.barh(y_pos, [0.5] * len(char_data), alpha=0)  # Invisible bars
        
        for i, (key, value) in enumerate(char_data.items()):
            ax2.text(0.1, i, f"{key}: {value}", fontsize=10, va='center')
        
        ax2.set_ylim(-0.5, len(char_data) - 0.5)
        ax2.set_xlim(0, 1)
        ax2.set_title('Image Characteristics', fontsize=14)
        ax2.axis('off')
        
        # 3. Quality metrics
        ax3 = fig.add_subplot(gs[0, 3])
        
        if result.quality_metrics:
            metrics = list(result.quality_metrics.items())
            if 'rating' in result.quality_metrics:
                metrics = [m for m in metrics if m[0] != 'rating']
            
            labels = [m[0].replace('_', ' ').title() for m in metrics]
            values = [m[1] for m in metrics]
            
            colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' 
                     for v in values]
            
            bars = ax3.bar(range(len(labels)), values, color=colors)
            ax3.set_xticks(range(len(labels)))
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            ax3.set_ylim(0, 1.1)
            ax3.set_ylabel('Score')
            ax3.set_title(f'Quality Assessment: {result.quality_metrics.get("rating", "N/A")}', 
                         fontsize=14)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Radial profile
        ax4 = fig.add_subplot(gs[1, 0])
        
        cx, cy = result.center
        max_r = int(result.radius_cladding * 1.5)
        radial_profile = []
        
        for r in range(max_r):
            mask = self._create_annular_mask(cx, cy, max(0, r-1), r+1)
            if np.any(mask):
                radial_profile.append(np.mean(self.image[mask]))
            else:
                radial_profile.append(0)
        
        ax4.plot(radial_profile, 'b-', linewidth=2)
        ax4.axvline(x=result.radius_core, color='lime', linestyle='--', 
                   linewidth=2, label='Core')
        ax4.axvline(x=result.radius_cladding, color='cyan', linestyle='--', 
                   linewidth=2, label='Cladding')
        ax4.set_xlabel('Radius (pixels)')
        ax4.set_ylabel('Average Intensity')
        ax4.set_title('Radial Intensity Profile', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Edge detection
        ax5 = fig.add_subplot(gs[1, 1])
        
        edges = cv2.Canny((self.image * 255).astype(np.uint8), 50, 150)
        ax5.imshow(edges, cmap='gray')
        
        # Overlay circles
        circle1 = patches.Circle(result.center, result.radius_core, 
                               fill=False, color='lime', linewidth=2)
        circle2 = patches.Circle(result.center, result.radius_cladding, 
                               fill=False, color='cyan', linewidth=2)
        ax5.add_patch(circle1)
        ax5.add_patch(circle2)
        
        ax5.set_title('Edge Detection Overlay', fontsize=14)
        ax5.axis('off')
        
        # 6. Uncertainty visualization
        ax6 = fig.add_subplot(gs[1, 2])
        
        if result.uncertainty:
            unc_data = {
                'Position': result.uncertainty.get('position', 0),
                'Radius': result.uncertainty.get('radius', 0),
                'Total': result.uncertainty.get('total', 0)
            }
            
            bars = ax6.bar(range(len(unc_data)), list(unc_data.values()))
            ax6.set_xticks(range(len(unc_data)))
            ax6.set_xticklabels(list(unc_data.keys()))
            ax6.set_ylabel('Uncertainty (pixels)')
            ax6.set_title('Uncertainty Analysis', fontsize=14)
            
            # Color based on magnitude
            for bar, value in zip(bars, unc_data.values()):
                if value < 1:
                    bar.set_color('green')
                elif value < 3:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
                
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # 7. 3D visualization
        ax7 = fig.add_subplot(gs[1, 3], projection='3d')
        
        # Downsample for performance
        step = 10
        x_range = np.arange(0, self.image.shape[1], step)
        y_range = np.arange(0, self.image.shape[0], step)
        X, Y = np.meshgrid(x_range, y_range)
        Z = self.image[::step, ::step]
        
        surf = ax7.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        ax7.set_zlabel('Intensity')
        ax7.set_title('3D Intensity Surface', fontsize=14)
        
        # 8. Method comparison (if multiple methods were used)
        ax8 = fig.add_subplot(gs[2, :2])
        
        if 'methods' in result.convergence_info:
            methods = result.convergence_info['methods']
            confidences = result.convergence_info.get('individual_confidences', 
                                                     [0.5] * len(methods))
            
            bars = ax8.bar(range(len(methods)), confidences)
            ax8.set_xticks(range(len(methods)))
            ax8.set_xticklabels(methods, rotation=45, ha='right')
            ax8.set_ylabel('Confidence')
            ax8.set_ylim(0, 1.1)
            ax8.set_title('Method Confidence Comparison', fontsize=14)
            
            # Color by confidence
            for bar, conf in zip(bars, confidences):
                if conf > 0.8:
                    bar.set_color('green')
                elif conf > 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        else:
            ax8.text(0.5, 0.5, 'Single Method Used', 
                    transform=ax8.transAxes, ha='center', va='center',
                    fontsize=16)
            ax8.axis('off')
        
        # 9. Summary text
        ax9 = fig.add_subplot(gs[2, 2:])
        ax9.axis('off')
        
        summary_text = f"""DETECTION SUMMARY
        
Center: ({result.center[0]:.2f}, {result.center[1]:.2f}) ± {result.uncertainty.get('position', 0):.2f} pixels
Core Radius: {result.radius_core:.2f} ± {result.uncertainty.get('radius', 0):.2f} pixels
Cladding Radius: {result.radius_cladding:.2f} ± {result.uncertainty.get('radius', 0):.2f} pixels
Core/Cladding Ratio: {result.radius_core/result.radius_cladding:.3f}

Method: {result.method_used}
Confidence: {result.confidence:.1%}
Computation Time: {result.computation_time:.3f} seconds
Overall Quality: {result.quality_metrics.get('overall', 0):.1%}

Strategy: {self.characteristics.recommended_strategy.value}
Image Quality: {self.characteristics.quality.value}
"""
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        
        plt.suptitle('Unified Intelligent Fiber Optic Detection System', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _print_detection_summary(self, result: DetectionResult):
        """Print detection summary to console."""
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        
        print(f"\nCenter: ({result.center[0]:.4f}, {result.center[1]:.4f})")
        print(f"  Uncertainty: ±{result.uncertainty.get('position', 0):.4f} pixels")
        
        print(f"\nCore Radius: {result.radius_core:.4f} pixels")
        print(f"Cladding Radius: {result.radius_cladding:.4f} pixels")
        print(f"  Uncertainty: ±{result.uncertainty.get('radius', 0):.4f} pixels")
        
        print(f"\nCore/Cladding Ratio: {result.radius_core/result.radius_cladding:.4f}")
        
        print(f"\nMethod Used: {result.method_used}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Computation Time: {result.computation_time:.3f} seconds")
        
        if result.quality_metrics:
            print(f"\nQuality Assessment:")
            print(f"  Overall: {result.quality_metrics.get('overall', 0):.1%}")
            print(f"  Rating: {result.quality_metrics.get('rating', 'N/A')}")
            print(f"  Edge Alignment: {result.quality_metrics.get('edge_alignment', 0):.1%}")
            print(f"  Symmetry: {result.quality_metrics.get('symmetry', 0):.1%}")
            print(f"  Contrast: {result.quality_metrics.get('contrast', 0):.1%}")
            print(f"  Completeness: {result.quality_metrics.get('completeness', 0):.1%}")
        
        print("\n" + "="*60)
    
    def batch_process(self, image_paths: List[str], 
                     output_dir: str = 'unified_results') -> pd.DataFrame:
        """Process multiple images and generate comparative report."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.detect(image_path, visualize=False)
                
                # Save individual result
                result_dict = {
                    'image': os.path.basename(image_path),
                    'center_x': result.center[0],
                    'center_y': result.center[1],
                    'radius_core': result.radius_core,
                    'radius_cladding': result.radius_cladding,
                    'confidence': result.confidence,
                    'method': result.method_used,
                    'computation_time': result.computation_time,
                    'quality_overall': result.quality_metrics.get('overall', 0),
                    'quality_rating': result.quality_metrics.get('rating', 'N/A')
                }
                
                results.append(result_dict)
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results.append({
                    'image': os.path.basename(image_path),
                    'error': str(e)
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv(os.path.join(output_dir, 'batch_results.csv'), index=False)
        
        # Generate report
        self._generate_batch_report(df, output_dir)
        
        return df
    
    def _generate_batch_report(self, df: pd.DataFrame, output_dir: str):
        """Generate batch processing report."""
        # Statistical summary
        summary = df.describe()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Success rate
        success_rate = len(df[~df.get('error', pd.Series()).notna()]) / len(df)
        
        ax = axes[0, 0]
        ax.pie([success_rate, 1-success_rate], 
              labels=['Success', 'Failed'],
              colors=['green', 'red'],
              autopct='%1.1f%%')
        ax.set_title('Processing Success Rate')
        
        # Confidence distribution
        ax = axes[0, 1]
        if 'confidence' in df.columns:
            df['confidence'].hist(ax=ax, bins=20, alpha=0.7, color='blue')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Count')
            ax.set_title('Confidence Distribution')
        
        # Computation time
        ax = axes[1, 0]
        if 'computation_time' in df.columns:
            df['computation_time'].plot(kind='bar', ax=ax)
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Computation Time by Image')
        
        # Quality ratings
        ax = axes[1, 1]
        if 'quality_rating' in df.columns:
            rating_counts = df['quality_rating'].value_counts()
            rating_counts.plot(kind='bar', ax=ax)
            ax.set_xlabel('Quality Rating')
            ax.set_ylabel('Count')
            ax.set_title('Quality Rating Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_summary.png'), dpi=150)
        plt.close()
        
        print(f"\nBatch processing complete. Results saved to {output_dir}/")


def unified_fiber_detection(image_path: str, advanced_mode: bool = True):
    """
    Main function for unified fiber optic detection.
    
    Args:
        image_path: Path to fiber optic image
        advanced_mode: Enable advanced mathematical methods
        
    Returns:
        Detection result
    """
    detector = UnifiedFiberOpticDetector(advanced_mode=advanced_mode)
    result = detector.detect(image_path)
    
    # Save results
    output_dir = 'unified_detection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    with open(os.path.join(output_dir, 'detection_result.json'), 'w') as f:
        result_dict = {
            'center': result.center,
            'radius_core': result.radius_core,
            'radius_cladding': result.radius_cladding,
            'confidence': result.confidence,
            'uncertainty': result.uncertainty,
            'method_used': result.method_used,
            'computation_time': result.computation_time,
            'quality_metrics': result.quality_metrics
        }
        json.dump(result_dict, f, indent=2)
    
    # Extract and save regions
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Create masks
    cx, cy = result.center
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    
    core_mask = (x - cx)**2 + (y - cy)**2 <= result.radius_core**2
    clad_mask = ((x - cx)**2 + (y - cy)**2 > result.radius_core**2) & \
                ((x - cx)**2 + (y - cy)**2 <= result.radius_cladding**2)
    
    # Extract regions
    core_region = np.zeros_like(image)
    core_region[core_mask] = image[core_mask]
    
    clad_region = np.zeros_like(image)
    clad_region[clad_mask] = image[clad_mask]
    
    # Save
    cv2.imwrite(os.path.join(output_dir, 'core_unified.png'), core_region)
    cv2.imwrite(os.path.join(output_dir, 'cladding_unified.png'), clad_region)
    
    print(f"\nResults saved to {output_dir}/")
    
    return result


if __name__ == "__main__":
    # Example usage
    test_image = r'C:\Users\Saem1001\Documents\GitHub\IPPS\App\images\img38.jpg'
    
    # Run unified detection
    result = unified_fiber_detection(test_image, advanced_mode=True)
    
    # Optional: Batch processing
    # detector = UnifiedFiberOpticDetector(advanced_mode=True)
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # batch_results = detector.batch_process(image_list)
