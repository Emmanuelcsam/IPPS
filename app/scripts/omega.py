#!/usr/bin/env python3
"""
OMEGA: Self-Improving Fiber Optic Analysis Framework
=====================================================
This script runs *after* daniel.py, using its output to
provide advanced analysis and learning capabilities.

Methodologies:
1.  **Golden Template Comparison:** Compares images against a "perfect"
    reference for robust, simple defect detection.
2.  **ML-Based Classification:** Uses a trained classifier to identify
    defect types based on extracted features.
3.  **Anomaly Detection:** Employs an autoencoder to find unusual defects
    not seen during training.
4.  **Feedback Loop:** Includes a mechanism to review and correct
    analyses, feeding data back to improve the models.

Author: Self-Improving Systems Division
Version: 1.0
"""

import os
import cv2
import numpy as np
import argparse
import json
import glob
from tqdm import tqdm
import pandas as pd
import pickle

from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuration ---
# Ensure this script can find the daniel script to import from it
import sys
# Add the scripts directory to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from daniel import EnhancedDefectDetector, DefectType, create_realistic_fiber_image

# --- Directory and File Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

GOLDEN_TEMPLATE_DIR = os.path.join(DATA_DIR, 'golden_template')
GOLDEN_TEMPLATE_PATH = os.path.join(GOLDEN_TEMPLATE_DIR, 'golden_template.png')
GOOD_IMAGES_DIR = os.path.join(DATA_DIR, 'raw_images', 'good')
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'training_data', 'defect_features.csv')

CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'defect_classifier.pkl')
AUTOENCODER_PATH = os.path.join(MODELS_DIR, 'autoencoder_model.h5')

# --- Helper Functions ---
def setup_directories():
    """Creates the necessary directories for the project."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GOLDEN_TEMPLATE_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'raw_images', 'good'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'raw_images', 'defective'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'training_data'), exist_ok=True)
    print("✓ Project directories are set up.")

def preprocess_for_template(image_path, size=(512, 512)):
    """Loads and preprocesses an image for template creation or comparison."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Use the same robust preprocessing from daniel.py
    detector = EnhancedDefectDetector()
    processed_img = detector._preprocess_image_advanced(img)

    # Resize to a standard size for consistency
    resized_img = cv2.resize(processed_img, size, interpolation=cv2.INTER_AREA)
    
    # Isolate the fiber itself using a mask
    masks, localization = detector._generate_fiber_masks_enhanced(resized_img)
    if not masks:
        masks, localization = detector._fallback_mask_generation(resized_img)
        
    if masks and 'Fiber' in masks:
        fiber_only = cv2.bitwise_and(resized_img, resized_img, mask=masks['Fiber'])
        return fiber_only
    return resized_img # Fallback to the full preprocessed image

# --- Core Methodologies ---

## 1. Golden Template Method
def create_golden_template(args):
    """
    Averages multiple high-quality, defect-free images to create a
    "golden template" reference image.
    """
    print("Creating Golden Template...")
    image_paths = glob.glob(os.path.join(args.good_images_dir, '*'))
    if not image_paths:
        print(f"✗ Error: No images found in {args.good_images_dir}. Cannot create template.")
        # As a fallback, create a synthetic good image
        print("Creating a synthetic 'good' image as a template.")
        good_image = create_realistic_fiber_image(size=512, defect_complexity="none")
        cv2.imwrite(GOLDEN_TEMPLATE_PATH, good_image)
        print(f"✓ Syntethic golden template saved to {GOLDEN_TEMPLATE_PATH}")
        return

    templates = []
    for path in tqdm(image_paths, desc="Processing good images"):
        processed = preprocess_for_template(path)
        if processed is not None:
            templates.append(processed)

    if not templates:
        print("✗ Error: Could not process any images to create a template.")
        return

    # Calculate the average of all the good templates
    golden_template = np.mean(templates, axis=0).astype(np.uint8)
    
    cv2.imwrite(GOLDEN_TEMPLATE_PATH, golden_template)
    print(f"✓ Golden template created and saved to {GOLDEN_TEMPLATE_PATH}")

## 2. ML Classifier
def train_classifier(args):
    """
    Trains a feature-based classifier (Random Forest) on labeled defect data.
    """
    print("Training Defect Classifier...")
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"✗ Error: Training data not found at {TRAINING_DATA_PATH}.")
        print("  Run the 'review-and-learn' command to generate training data from analysis reports.")
        return

    df = pd.read_csv(TRAINING_DATA_PATH)
    if len(df) < 10:
        print(f"✗ Error: Not enough training data. Found only {len(df)} samples.")
        return

    # Drop non-feature columns
    labels = df['true_defect_type']
    features = df.drop(columns=['image_path', 'defect_id', 'true_defect_type'])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    print("\nClassifier Performance Report:")
    print(classification_report(y_test, predictions))

    # Save the trained model
    with open(CLASSIFIER_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Defect classifier model saved to {CLASSIFIER_PATH}")

## 3. Anomaly Detection Autoencoder
def get_autoencoder(input_shape=(256, 256, 1)):
    """Defines the Convolutional Autoencoder architecture."""
    input_img = Input(shape=input_shape)
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(args):
    """
    Trains a convolutional autoencoder on "good" images only to learn
    the representation of a normal fiber end-face.
    """
    print("Training Anomaly Detection Autoencoder...")
    image_paths = glob.glob(os.path.join(args.good_images_dir, '*'))
    if not image_paths:
        print(f"✗ Error: No 'good' images found in {args.good_images_dir} to train the autoencoder.")
        return
        
    # Load and preprocess images
    train_images = []
    size = (256, 256)
    for path in tqdm(image_paths, desc="Loading good images for autoencoder"):
        img = preprocess_for_template(path, size=size)
        if img is not None:
            # Normalize to [0, 1] for the autoencoder
            train_images.append(img.astype('float32') / 255.)

    if not train_images:
        print("✗ Error: Failed to load any valid images for training.")
        return

    train_images = np.array(train_images).reshape(-1, size[0], size[1], 1)

    X_train, X_val = train_test_split(train_images, test_size=0.1, random_state=42)

    autoencoder = get_autoencoder(input_shape=(size[0], size[1], 1))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    print(f"Training autoencoder on {len(X_train)} images...")
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=16,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=callbacks
    )

    autoencoder.save(AUTOENCODER_PATH)
    print(f"✓ Autoencoder model saved to {AUTOENCODER_PATH}")

## 4. Main Analysis and Feedback Loop
def review_and_learn(args):
    """
    Processes analysis reports from daniel.py, extracts features,
    and creates a labeled dataset for training the classifier.
    This simulates a human-in-the-loop review process.
    """
    print("Reviewing analysis reports and building training data...")
    report_paths = glob.glob(os.path.join(args.reports_dir, '*.json'))
    if not report_paths:
        print(f"✗ Error: No analysis reports (.json) found in {args.reports_dir}.")
        return

    all_features = []
    for report_path in tqdm(report_paths, desc="Processing reports"):
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        image_path = report['image_info']['path']
        for defect in report['defects']:
            # In a real system, you'd show the defect to a user for labeling.
            # Here, we'll use the classification from daniel.py as the "true" label
            # for demonstration purposes.
            true_label = defect['type']
            
            features = defect['features']
            features['image_path'] = image_path
            features['defect_id'] = defect['id']
            features['true_defect_type'] = true_label
            all_features.append(features)

    if not all_features:
        print("✗ No defects found in any reports to generate training data.")
        return

    df = pd.DataFrame(all_features)
    df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"✓ Training data created/updated at {TRAINING_DATA_PATH}")
    print(f"  Found {len(df)} total defects. You can now train the classifier.")

def analyze_image(args):
    """
    Runs a full analysis on a new image using all methodologies.
    """
    image_path = args.image_path
    print(f"🔬 Running OMEGA analysis on: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        print(f"✗ Error: Image file not found at {image_path}")
        return

    # --- Setup ---
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    analysis_results = {
        "image_path": image_path,
        "golden_template_defects": [],
        "ml_classifier_predictions": [],
        "autoencoder_anomalies": []
    }

    # --- 1. Golden Template Comparison ---
    if os.path.exists(GOLDEN_TEMPLATE_PATH):
        print("\n--- Method 1: Golden Template Comparison ---")
        golden_template = cv2.imread(GOLDEN_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
        test_image_processed = preprocess_for_template(image_path, size=golden_template.shape)
        
        (score, diff) = ssim(golden_template, test_image_processed, full=True)
        diff = (diff * 255).astype("uint8")
        
        _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_image = original_image.copy()
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 20: # Min area filter
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output_image, "Template Mismatch", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                analysis_results["golden_template_defects"].append({"id": i, "bbox": [x,y,w,h]})
        
        print(f"Found {len(analysis_results['golden_template_defects'])} potential defects.")
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'golden_template_analysis.png'), output_image)
        print(f"✓ Golden template result saved to {OUTPUT_DIR}")
    else:
        print(" Golden Template model not found. Skipping.")


    # --- 2. ML Classifier Prediction ---
    # First, run daniel.py to get initial defect data
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    report_filename = f"analysis_report_{base_filename}.json"
    if not os.path.exists(report_filename):
         print(f"\n--- Prerequisite: Running daniel.py analysis to get initial defects ---")
         # This assumes daniel.py is runnable from the command line
         os.system(f"python {os.path.join(os.path.dirname(__file__), 'daniel.py')} {image_path}")
    
    if os.path.exists(CLASSIFIER_PATH) and os.path.exists(report_filename):
        print("\n--- Method 2: ML Classifier Prediction ---")
        with open(CLASSIFIER_PATH, 'rb') as f:
            classifier = pickle.load(f)
        
        with open(report_filename, 'r') as f:
            report = json.load(f)

        output_image = original_image.copy()
        print("Re-classifying defects from daniel.py report...")
        for defect in report['defects']:
            # Create a dataframe from the features for prediction
            features_df = pd.DataFrame([defect['features']])
            
            # Ensure columns match training columns (handle missing columns)
            with open(CLASSIFIER_PATH, 'rb') as f_model:
                model_columns = pickle.load(f_model).feature_names_in_
            features_df = features_df.reindex(columns=model_columns, fill_value=0)

            prediction = classifier.predict(features_df)[0]
            
            x, y, w, h = defect['bbox']
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = f"ML Prediction: {prediction}"
            cv2.putText(output_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            analysis_results["ml_classifier_predictions"].append({
                "id": defect['id'],
                "original_type": defect['type'],
                "predicted_type": prediction,
                "bbox": defect['bbox']
            })
        
        print(f"Predicted types for {len(analysis_results['ml_classifier_predictions'])} defects.")
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'ml_classifier_analysis.png'), output_image)
        print(f"✓ ML Classifier result saved to {OUTPUT_DIR}")
    else:
        print("\n--- Method 2: ML Classifier Prediction ---")
        print(" ML Classifier or report not found. Skipping.")


    # --- 3. Autoencoder Anomaly Detection ---
    if os.path.exists(AUTOENCODER_PATH):
        print("\n--- Method 3: Autoencoder Anomaly Detection ---")
        from tensorflow.keras.models import load_model
        autoencoder = load_model(AUTOENCODER_PATH)
        
        # Preprocess and resize
        size = (256, 256)
        test_image_processed = preprocess_for_template(image_path, size=size)
        test_image_normalized = test_image_processed.astype('float32') / 255.
        test_image_reshaped = np.reshape(test_image_normalized, (1, size[0], size[1], 1))

        # Get reconstruction
        reconstruction = autoencoder.predict(test_image_reshaped)
        
        # Calculate reconstruction error
        error = np.square(test_image_reshaped - reconstruction).mean(axis=-1).squeeze()
        
        # Threshold the error map to find anomalies
        error_threshold = np.quantile(error, 0.98) # Highlight top 2% of errors
        anomaly_map = (error > error_threshold).astype("uint8") * 255
        
        # Upscale anomaly map to original image size
        anomaly_map_resized = cv2.resize(anomaly_map, (gray_image.shape[1], gray_image.shape[0]))

        contours, _ = cv2.findContours(anomaly_map_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_image = original_image.copy()
        for i, cnt in enumerate(contours):
             if cv2.contourArea(cnt) > 30: # Min area filter
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output_image, "Anomaly", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                analysis_results["autoencoder_anomalies"].append({"id": i, "bbox": [x,y,w,h]})

        print(f"Found {len(analysis_results['autoencoder_anomalies'])} anomalies.")
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'autoencoder_analysis.png'), output_image)
        print(f"✓ Autoencoder result saved to {OUTPUT_DIR}")
    else:
        print("\n--- Method 3: Autoencoder Anomaly Detection ---")
        print(" Autoencoder model not found. Skipping.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OMEGA: Self-Improving Fiber Optic Analysis Framework.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')
    
    # Command: setup
    parser_setup = subparsers.add_parser('setup', help='Create the necessary project directories.')
    parser_setup.set_defaults(func=lambda args: setup_directories())

    # Command: create-template
    parser_template = subparsers.add_parser('create-template', help='Build the golden template from good images.')
    parser_template.add_argument('--good-images-dir', type=str, default=GOOD_IMAGES_DIR, help='Directory containing defect-free images.')
    parser_template.set_defaults(func=create_golden_template)

    # Command: review-and-learn
    parser_learn = subparsers.add_parser('review-and-learn', help='Generate training data from daniel.py reports.')
    parser_learn.add_argument('--reports-dir', type=str, default='.', help='Directory containing the JSON analysis reports.')
    parser_learn.set_defaults(func=review_and_learn)

    # Command: train-classifier
    parser_train_cls = subparsers.add_parser('train-classifier', help='Train the ML defect classifier.')
    parser_train_cls.set_defaults(func=train_classifier)

    # Command: train-autoencoder
    parser_train_ae = subparsers.add_parser('train-autoencoder', help='Train the anomaly detection autoencoder.')
    parser_train_ae.add_argument('--good-images-dir', type=str, default=GOOD_IMAGES_DIR, help='Directory containing defect-free images.')
    parser_train_ae.set_defaults(func=train_autoencoder)

    # Command: analyze
    parser_analyze = subparsers.add_parser('analyze', help='Run the full OMEGA analysis on a single image.')
    parser_analyze.add_argument('image_path', type=str, help='Path to the fiber optic image to analyze.')
    parser_analyze.set_defaults(func=analyze_image)
    
    args = parser.parse_args()
    args.func(args)
