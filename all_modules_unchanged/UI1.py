#!/usr/bin/env python3
"""
Image Processing Pipeline UI
============================
A comprehensive UI for applying image processing functions in customizable pipelines.
Similar to Gwyddion but for fiber optic image analysis.

Author: Assistant
Date: 2025
"""

import sys
import os
import cv2
import numpy as np
import importlib.util
import inspect
import ast
from pathlib import Path
from datetime import datetime
import json
import traceback
from typing import Dict, List, Tuple, Any, Optional, Callable

# Qt imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QListWidget, QListWidgetItem, QSplitter, QTextEdit,
                           QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QGroupBox, QScrollArea, QTableWidget,
                           QTableWidgetItem, QHeaderView, QMessageBox,
                           QProgressBar, QSlider, QMenu, QAction, QToolBar,
                           QDockWidget, QTabWidget, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread, QTimer, QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon

class ImageViewer(QLabel):
    """Custom image viewer with zoom and pan functionality"""
    
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.displayed_image = None
        self.zoom_factor = 1.0
        self.pan_start = None
        self.pan_offset = QPoint(0, 0)
        self.setMouseTracking(True)
        self.setScaledContents(False)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        
    def set_image(self, image):
        """Set the image to display"""
        if image is None:
            self.clear()
            return
            
        # Convert to RGB for display
        if len(image.shape) == 2:
            # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # Color - convert BGR to RGB
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self.original_image = QPixmap.fromImage(q_image)
        self.update_display()
        
    def update_display(self):
        """Update the displayed image with current zoom and pan"""
        if self.original_image is None:
            return
            
        # Calculate scaled size
        scaled_width = int(self.original_image.width() * self.zoom_factor)
        scaled_height = int(self.original_image.height() * self.zoom_factor)
        
        # Scale the image
        scaled_pixmap = self.original_image.scaled(
            scaled_width, scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Apply pan offset
        painter = QPainter(self)
        self.displayed_image = QPixmap(self.size())
        self.displayed_image.fill(Qt.white)
        
        painter = QPainter(self.displayed_image)
        painter.drawPixmap(self.pan_offset, scaled_pixmap)
        painter.end()
        
        self.setPixmap(self.displayed_image)
        
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_image is None:
            return
            
        # Get the position of the mouse relative to the widget
        pos = event.pos()
        
        # Calculate zoom
        delta = event.angleDelta().y()
        zoom_in = delta > 0
        zoom_speed = 1.1
        
        old_zoom = self.zoom_factor
        if zoom_in:
            self.zoom_factor *= zoom_speed
        else:
            self.zoom_factor /= zoom_speed
            
        # Limit zoom
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        
        # Adjust pan to zoom around mouse position
        if old_zoom != self.zoom_factor:
            zoom_ratio = self.zoom_factor / old_zoom
            self.pan_offset = pos - zoom_ratio * (pos - self.pan_offset)
            
        self.update_display()
        
    def mousePressEvent(self, event):
        """Start panning"""
        if event.button() == Qt.LeftButton:
            self.pan_start = event.pos()
            
    def mouseMoveEvent(self, event):
        """Handle panning"""
        if self.pan_start is not None:
            delta = event.pos() - self.pan_start
            self.pan_offset += delta
            self.pan_start = event.pos()
            self.update_display()
            
    def mouseReleaseEvent(self, event):
        """Stop panning"""
        if event.button() == Qt.LeftButton:
            self.pan_start = None
            
    def reset_view(self):
        """Reset zoom and pan"""
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update_display()


class FunctionScanner:
    """Scans Python files and extracts callable functions"""
    
    def __init__(self, scripts_dir):
        self.scripts_dir = Path(scripts_dir)
        self.functions = {}
        self.scan_scripts()
        
    def scan_scripts(self):
        """Scan all Python files in the scripts directory"""
        if not self.scripts_dir.exists():
            print(f"Scripts directory not found: {self.scripts_dir}")
            return
            
        for script_path in self.scripts_dir.glob("*.py"):
            try:
                self.extract_functions(script_path)
            except Exception as e:
                print(f"Error scanning {script_path}: {e}")
                
    def extract_functions(self, script_path):
        """Extract functions from a Python file"""
        # Read the file content
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse the AST
        try:
            tree = ast.parse(content)
        except:
            return
            
        # Extract function information
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions and main
                if node.name.startswith('_') or node.name == 'main':
                    continue
                    
                func_info = {
                    'name': node.name,
                    'file': script_path.name,
                    'path': str(script_path),
                    'docstring': ast.get_docstring(node) or "",
                    'args': [arg.arg for arg in node.args.args],
                    'category': self.categorize_function(script_path.name, node.name)
                }
                
                # Store by a unique key
                key = f"{script_path.stem}.{node.name}"
                self.functions[key] = func_info
                
    def categorize_function(self, filename, func_name):
        """Categorize function based on filename and function name"""
        filename_lower = filename.lower()
        func_lower = func_name.lower()
        
        if 'threshold' in filename_lower or 'threshold' in func_lower:
            return "Thresholding"
        elif 'blur' in filename_lower or 'filter' in filename_lower:
            return "Filtering"
        elif 'edge' in filename_lower or 'canny' in func_lower or 'sobel' in func_lower:
            return "Edge Detection"
        elif 'morph' in filename_lower:
            return "Morphology"
        elif 'circle' in filename_lower or 'hough' in func_lower:
            return "Circle Detection"
        elif 'mask' in filename_lower:
            return "Masking"
        elif 'enhance' in filename_lower or 'clahe' in func_lower:
            return "Enhancement"
        elif 'defect' in filename_lower:
            return "Defect Detection"
        elif 'color' in filename_lower or 'heatmap' in func_lower:
            return "Colorization"
        elif 'save' in filename_lower or 'load' in filename_lower:
            return "I/O"
        else:
            return "Other"
            
    def search_functions(self, keyword):
        """Search functions by keyword"""
        keyword_lower = keyword.lower()
        results = {}
        
        for key, func_info in self.functions.items():
            if (keyword_lower in func_info['name'].lower() or
                keyword_lower in func_info['file'].lower() or
                keyword_lower in func_info['docstring'].lower() or
                keyword_lower in func_info['category'].lower()):
                results[key] = func_info
                
        return results


class FunctionWrapper:
    """Wraps script functions to make them pipeline-compatible"""
    
    @staticmethod
    def create_wrapper(script_path, function_name):
        """Create a wrapper for a function from a script"""
        def wrapper(image, **kwargs):
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location("module", script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get the function
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    
                    # Call the function
                    # Try different calling conventions
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    
                    if len(params) == 0:
                        # No parameters - might use global state
                        return image
                    elif len(params) == 1:
                        # Image only
                        result = func(image)
                    else:
                        # Image + additional parameters
                        result = func(image, **kwargs)
                        
                    return result
                else:
                    # If function not found, try to run the script's main processing
                    # Look for common patterns in the scripts
                    if hasattr(module, 'process_image'):
                        return module.process_image(image)
                    elif hasattr(module, 'apply_filter'):
                        return module.apply_filter(image)
                    else:
                        # Last resort - return unchanged
                        return image
                        
            except Exception as e:
                print(f"Error in wrapper for {function_name}: {e}")
                traceback.print_exc()
                return image
                
        return wrapper


class ProcessingPipeline:
    """Manages the image processing pipeline"""
    
    def __init__(self):
        self.steps = []
        self.current_image = None
        self.history = []
        self.history_index = -1
        
    def add_step(self, func_info, params=None):
        """Add a processing step to the pipeline"""
        self.steps.append({
            'function': func_info,
            'params': params or {},
            'enabled': True
        })
        
    def remove_step(self, index):
        """Remove a step from the pipeline"""
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
            
    def move_step(self, from_index, to_index):
        """Move a step in the pipeline"""
        if (0 <= from_index < len(self.steps) and 
            0 <= to_index < len(self.steps)):
            step = self.steps.pop(from_index)
            self.steps.insert(to_index, step)
            
    def toggle_step(self, index):
        """Enable/disable a step"""
        if 0 <= index < len(self.steps):
            self.steps[index]['enabled'] = not self.steps[index]['enabled']
            
    def execute(self, image, progress_callback=None):
        """Execute the pipeline on an image"""
        if image is None:
            return None
            
        result = image.copy()
        total_steps = len([s for s in self.steps if s['enabled']])
        completed = 0
        
        for step in self.steps:
            if not step['enabled']:
                continue
                
            try:
                # Create wrapper function
                func_info = step['function']
                wrapper = FunctionWrapper.create_wrapper(
                    func_info['path'], 
                    func_info['name']
                )
                
                # Execute step
                result = wrapper(result, **step['params'])
                
                # Update progress
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_steps)
                    
            except Exception as e:
                print(f"Error executing {func_info['name']}: {e}")
                traceback.print_exc()
                
        return result
        
    def save_pipeline(self, filepath):
        """Save pipeline configuration"""
        config = {
            'steps': [
                {
                    'function': step['function'],
                    'params': step['params'],
                    'enabled': step['enabled']
                }
                for step in self.steps
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
    def load_pipeline(self, filepath):
        """Load pipeline configuration"""
        with open(filepath, 'r') as f:
            config = json.load(f)
            
        self.steps = config['steps']


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.processed_image = None
        self.pipeline = ProcessingPipeline()
        self.function_scanner = None
        self.init_ui()
        self.load_functions()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Image Processing Pipeline")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set up central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Function browser
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel - Image viewer
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # Right panel - Pipeline
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([300, 800, 300])
        
        # Status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Image...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Result...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        load_pipeline_action = QAction('Load Pipeline...', self)
        load_pipeline_action.triggered.connect(self.load_pipeline)
        file_menu.addAction(load_pipeline_action)
        
        save_pipeline_action = QAction('Save Pipeline...', self)
        save_pipeline_action.triggered.connect(self.save_pipeline)
        file_menu.addAction(save_pipeline_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        undo_action = QAction('Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction('Redo', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut('Ctrl+0')
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        refresh_functions_action = QAction('Refresh Functions', self)
        refresh_functions_action.setShortcut('F5')
        refresh_functions_action.triggered.connect(self.load_functions)
        tools_menu.addAction(refresh_functions_action)
        
    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = self.addToolBar('Main')
        toolbar.setIconSize(QSize(24, 24))
        
        # Open image
        open_action = toolbar.addAction('Open')
        open_action.triggered.connect(self.open_image)
        
        # Save result
        save_action = toolbar.addAction('Save')
        save_action.triggered.connect(self.save_result)
        
        toolbar.addSeparator()
        
        # Execute pipeline
        execute_action = toolbar.addAction('Execute')
        execute_action.triggered.connect(self.execute_pipeline)
        
        # Clear pipeline
        clear_action = toolbar.addAction('Clear Pipeline')
        clear_action.triggered.connect(self.clear_pipeline)
        
        toolbar.addSeparator()
        
        # Zoom controls
        zoom_in_action = toolbar.addAction('Zoom In')
        zoom_in_action.triggered.connect(self.zoom_in)
        
        zoom_out_action = toolbar.addAction('Zoom Out')
        zoom_out_action.triggered.connect(self.zoom_out)
        
        reset_view_action = toolbar.addAction('Reset View')
        reset_view_action.triggered.connect(self.reset_view)
        
    def create_left_panel(self):
        """Create the left panel with function browser"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter keyword...")
        self.search_input.textChanged.connect(self.search_functions)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # Category filter
        category_layout = QHBoxLayout()
        category_label = QLabel("Category:")
        self.category_combo = QComboBox()
        self.category_combo.addItem("All")
        self.category_combo.currentTextChanged.connect(self.filter_by_category)
        category_layout.addWidget(category_label)
        category_layout.addWidget(self.category_combo)
        layout.addLayout(category_layout)
        
        # Function table
        self.function_table = QTableWidget()
        self.function_table.setColumnCount(3)
        self.function_table.setHorizontalHeaderLabels(["Function", "File", "Category"])
        self.function_table.horizontalHeader().setStretchLastSection(True)
        self.function_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.function_table.doubleClicked.connect(self.add_function_to_pipeline)
        layout.addWidget(self.function_table)
        
        # Function details
        details_group = QGroupBox("Function Details")
        details_layout = QVBoxLayout(details_group)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(150)
        details_layout.addWidget(self.details_text)
        layout.addWidget(details_group)
        
        # Add to pipeline button
        add_button = QPushButton("Add to Pipeline")
        add_button.clicked.connect(self.add_function_to_pipeline)
        layout.addWidget(add_button)
        
        # Connect selection change
        self.function_table.itemSelectionChanged.connect(self.show_function_details)
        
        return panel
        
    def create_center_panel(self):
        """Create the center panel with image viewer"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for original and processed images
        self.image_tabs = QTabWidget()
        
        # Original image viewer
        self.original_viewer = ImageViewer()
        self.image_tabs.addTab(self.original_viewer, "Original")
        
        # Processed image viewer
        self.processed_viewer = ImageViewer()
        self.image_tabs.addTab(self.processed_viewer, "Processed")
        
        layout.addWidget(self.image_tabs)
        
        # Image info
        info_layout = QHBoxLayout()
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        info_layout.addWidget(self.image_info_label)
        layout.addLayout(info_layout)
        
        return panel
        
    def create_right_panel(self):
        """Create the right panel with pipeline editor"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Pipeline label
        pipeline_label = QLabel("Processing Pipeline")
        pipeline_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(pipeline_label)
        
        # Pipeline list
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.pipeline_list)
        
        # Pipeline controls
        controls_layout = QGridLayout()
        
        move_up_button = QPushButton("Move Up")
        move_up_button.clicked.connect(self.move_step_up)
        controls_layout.addWidget(move_up_button, 0, 0)
        
        move_down_button = QPushButton("Move Down")
        move_down_button.clicked.connect(self.move_step_down)
        controls_layout.addWidget(move_down_button, 0, 1)
        
        toggle_button = QPushButton("Enable/Disable")
        toggle_button.clicked.connect(self.toggle_step)
        controls_layout.addWidget(toggle_button, 1, 0)
        
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_step)
        controls_layout.addWidget(remove_button, 1, 1)
        
        layout.addLayout(controls_layout)
        
        # Execute button
        execute_button = QPushButton("Execute Pipeline")
        execute_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        execute_button.clicked.connect(self.execute_pipeline)
        layout.addWidget(execute_button)
        
        # Clear pipeline button
        clear_button = QPushButton("Clear Pipeline")
        clear_button.clicked.connect(self.clear_pipeline)
        layout.addWidget(clear_button)
        
        # Add stretch
        layout.addStretch()
        
        return panel
        
    def load_functions(self):
        """Load functions from scripts directory"""
        # Get scripts directory (same directory as this script)
        scripts_dir = Path(__file__).parent
        
        # Scan for functions
        self.function_scanner = FunctionScanner(scripts_dir)
        
        # Populate category combo box
        categories = set()
        for func_info in self.function_scanner.functions.values():
            categories.add(func_info['category'])
        
        self.category_combo.clear()
        self.category_combo.addItem("All")
        for category in sorted(categories):
            self.category_combo.addItem(category)
            
        # Populate function table
        self.populate_function_table()
        
        self.status_bar.showMessage(f"Loaded {len(self.function_scanner.functions)} functions", 3000)
        
    def populate_function_table(self, functions=None):
        """Populate the function table"""
        if functions is None:
            functions = self.function_scanner.functions
            
        self.function_table.setRowCount(len(functions))
        
        row = 0
        for key, func_info in functions.items():
            # Function name
            name_item = QTableWidgetItem(func_info['name'])
            name_item.setData(Qt.UserRole, key)
            self.function_table.setItem(row, 0, name_item)
            
            # File name
            file_item = QTableWidgetItem(func_info['file'])
            self.function_table.setItem(row, 1, file_item)
            
            # Category
            category_item = QTableWidgetItem(func_info['category'])
            self.function_table.setItem(row, 2, category_item)
            
            row += 1
            
        self.function_table.resizeColumnsToContents()
        
    def search_functions(self, text):
        """Search functions based on input text"""
        if not text:
            self.populate_function_table()
            return
            
        results = self.function_scanner.search_functions(text)
        self.populate_function_table(results)
        
    def filter_by_category(self, category):
        """Filter functions by category"""
        if category == "All":
            self.populate_function_table()
            return
            
        filtered = {
            key: func_info 
            for key, func_info in self.function_scanner.functions.items()
            if func_info['category'] == category
        }
        
        self.populate_function_table(filtered)
        
    def show_function_details(self):
        """Show details of selected function"""
        current_row = self.function_table.currentRow()
        if current_row < 0:
            return
            
        name_item = self.function_table.item(current_row, 0)
        if name_item:
            key = name_item.data(Qt.UserRole)
            func_info = self.function_scanner.functions.get(key)
            
            if func_info:
                details = f"Function: {func_info['name']}\n"
                details += f"File: {func_info['file']}\n"
                details += f"Category: {func_info['category']}\n"
                details += f"Arguments: {', '.join(func_info['args'])}\n\n"
                details += f"Description:\n{func_info['docstring']}"
                
                self.details_text.setText(details)
                
    def add_function_to_pipeline(self):
        """Add selected function to pipeline"""
        current_row = self.function_table.currentRow()
        if current_row < 0:
            return
            
        name_item = self.function_table.item(current_row, 0)
        if name_item:
            key = name_item.data(Qt.UserRole)
            func_info = self.function_scanner.functions.get(key)
            
            if func_info:
                self.pipeline.add_step(func_info)
                self.update_pipeline_list()
                
    def update_pipeline_list(self):
        """Update the pipeline list widget"""
        self.pipeline_list.clear()
        
        for i, step in enumerate(self.pipeline.steps):
            func_info = step['function']
            text = f"{i+1}. {func_info['name']} ({func_info['file']})"
            
            if not step['enabled']:
                text = f"[DISABLED] {text}"
                
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)
            
            if not step['enabled']:
                item.setForeground(Qt.gray)
                
            self.pipeline_list.addItem(item)
            
    def move_step_up(self):
        """Move selected step up in pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row > 0:
            self.pipeline.move_step(current_row, current_row - 1)
            self.update_pipeline_list()
            self.pipeline_list.setCurrentRow(current_row - 1)
            
    def move_step_down(self):
        """Move selected step down in pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row < self.pipeline_list.count() - 1:
            self.pipeline.move_step(current_row, current_row + 1)
            self.update_pipeline_list()
            self.pipeline_list.setCurrentRow(current_row + 1)
            
    def toggle_step(self):
        """Toggle enabled state of selected step"""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            self.pipeline.toggle_step(current_row)
            self.update_pipeline_list()
            self.pipeline_list.setCurrentRow(current_row)
            
    def remove_step(self):
        """Remove selected step from pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            self.pipeline.remove_step(current_row)
            self.update_pipeline_list()
            
    def clear_pipeline(self):
        """Clear all steps from pipeline"""
        self.pipeline.steps.clear()
        self.update_pipeline_list()
        
    def open_image(self):
        """Open an image file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if filename:
            self.current_image = cv2.imread(filename)
            if self.current_image is not None:
                self.original_viewer.set_image(self.current_image)
                self.update_image_info()
                self.status_bar.showMessage(f"Loaded: {filename}", 3000)
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")
                
    def save_result(self):
        """Save the processed result"""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if filename:
            success = cv2.imwrite(filename, self.processed_image)
            if success:
                self.status_bar.showMessage(f"Saved: {filename}", 3000)
            else:
                QMessageBox.warning(self, "Error", "Failed to save image")
                
    def execute_pipeline(self):
        """Execute the processing pipeline"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
            
        if len(self.pipeline.steps) == 0:
            QMessageBox.warning(self, "Warning", "Pipeline is empty")
            return
            
        # Show progress
        self.progress_bar.show()
        self.progress_bar.setRange(0, len([s for s in self.pipeline.steps if s['enabled']]))
        
        def progress_callback(completed, total):
            self.progress_bar.setValue(completed)
            QApplication.processEvents()
            
        # Execute pipeline
        try:
            self.processed_image = self.pipeline.execute(
                self.current_image,
                progress_callback
            )
            
            # Show result
            if self.processed_image is not None:
                self.processed_viewer.set_image(self.processed_image)
                self.image_tabs.setCurrentIndex(1)  # Switch to processed tab
                self.status_bar.showMessage("Pipeline executed successfully", 3000)
            else:
                QMessageBox.warning(self, "Error", "Pipeline execution failed")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Pipeline execution error: {str(e)}")
            traceback.print_exc()
            
        finally:
            self.progress_bar.hide()
            
    def update_image_info(self):
        """Update image information display"""
        if self.current_image is None:
            self.image_info_label.setText("No image loaded")
            return
            
        height, width = self.current_image.shape[:2]
        channels = 1 if len(self.current_image.shape) == 2 else self.current_image.shape[2]
        dtype = self.current_image.dtype
        
        info = f"Size: {width}x{height} | Channels: {channels} | Type: {dtype}"
        self.image_info_label.setText(info)
        
    def zoom_in(self):
        """Zoom in the current viewer"""
        current_tab = self.image_tabs.currentIndex()
        if current_tab == 0:
            self.original_viewer.zoom_factor *= 1.2
            self.original_viewer.update_display()
        else:
            self.processed_viewer.zoom_factor *= 1.2
            self.processed_viewer.update_display()
            
    def zoom_out(self):
        """Zoom out the current viewer"""
        current_tab = self.image_tabs.currentIndex()
        if current_tab == 0:
            self.original_viewer.zoom_factor /= 1.2
            self.original_viewer.update_display()
        else:
            self.processed_viewer.zoom_factor /= 1.2
            self.processed_viewer.update_display()
            
    def reset_view(self):
        """Reset view for current viewer"""
        current_tab = self.image_tabs.currentIndex()
        if current_tab == 0:
            self.original_viewer.reset_view()
        else:
            self.processed_viewer.reset_view()
            
    def undo(self):
        """Undo last action"""
        # Implement undo functionality
        pass
        
    def redo(self):
        """Redo last undone action"""
        # Implement redo functionality
        pass
        
    def save_pipeline(self):
        """Save current pipeline configuration"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pipeline",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                self.pipeline.save_pipeline(filename)
                self.status_bar.showMessage(f"Pipeline saved: {filename}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save pipeline: {str(e)}")
                
    def load_pipeline(self):
        """Load pipeline configuration"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pipeline",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                self.pipeline.load_pipeline(filename)
                self.update_pipeline_list()
                self.status_bar.showMessage(f"Pipeline loaded: {filename}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load pipeline: {str(e)}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()