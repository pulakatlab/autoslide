"""
GUI for section viewing and validation.

This module provides a comprehensive interface for:
1. Filterable dataframe of all sections
2. Section image display with prediction overlay
3. Whole slide view with section highlights
4. Toggle-able section properties (validated, include)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json
from glob import glob
from ast import literal_eval
import cv2
import slideio
from tqdm import tqdm

from autoslide import config
from autoslide.utils.get_section_from_hash import (
    load_tracking_data, load_suggested_regions, get_section_from_hash
)
from autoslide.pipeline.utils import visualize_sections


class FilterableDataFrame:
    """A filterable dataframe widget using tkinter Treeview"""
    
    def __init__(self, parent, data, on_selection_change=None):
        self.parent = parent
        self.data = data.copy()
        self.filtered_data = data.copy()
        self.on_selection_change = on_selection_change
        
        self.setup_ui()
        self.populate_tree()
        
    def setup_ui(self):
        # Create main frame
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Filter frame
        filter_frame = ttk.Frame(self.frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_var.trace('w', self.on_filter_change)
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=30)
        filter_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(filter_frame, text="Column:").pack(side=tk.LEFT)
        self.filter_column = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_column, 
                                   values=["All"] + list(self.data.columns), width=15)
        filter_combo.pack(side=tk.LEFT, padx=(5, 10))
        filter_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        
        # Clear filter button
        ttk.Button(filter_frame, text="Clear", command=self.clear_filter).pack(side=tk.LEFT, padx=5)
        
        # Treeview frame with scrollbars
        tree_frame = ttk.Frame(self.frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview
        columns = list(self.data.columns)
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, minwidth=50)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        
    def populate_tree(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add filtered data
        for idx, row in self.filtered_data.iterrows():
            values = [str(row[col]) for col in self.data.columns]
            self.tree.insert('', tk.END, values=values, tags=(idx,))
    
    def on_filter_change(self, *args):
        filter_text = self.filter_var.get().lower()
        filter_col = self.filter_column.get()
        
        if not filter_text:
            self.filtered_data = self.data.copy()
        else:
            if filter_col == "All":
                # Search in all columns
                mask = self.data.astype(str).apply(
                    lambda x: x.str.lower().str.contains(filter_text, na=False)
                ).any(axis=1)
            else:
                # Search in specific column
                mask = self.data[filter_col].astype(str).str.lower().str.contains(filter_text, na=False)
            
            self.filtered_data = self.data[mask]
        
        self.populate_tree()
    
    def clear_filter(self):
        self.filter_var.set("")
        self.filter_column.set("All")
    
    def on_tree_select(self, event):
        selection = self.tree.selection()
        if selection and self.on_selection_change:
            item = self.tree.item(selection[0])
            tags = item['tags']
            if tags:
                original_idx = int(tags[0])
                selected_row = self.data.iloc[original_idx]
                self.on_selection_change(selected_row)
    
    def get_selected_row(self):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            tags = item['tags']
            if tags:
                original_idx = int(tags[0])
                return self.data.iloc[original_idx]
        return None
    
    def update_data(self, new_data):
        self.data = new_data.copy()
        self.filtered_data = new_data.copy()
        self.populate_tree()


class SectionViewer:
    """Main GUI application for section viewing and validation"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AutoSlide Section Viewer")
        self.root.geometry("1400x900")
        
        # Data
        self.sections_df = None
        self.current_section = None
        self.current_slide_scene = None
        self.current_slide_path = None
        
        # Images
        self.section_image = None
        self.section_mask = None
        self.slide_image = None
        
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for dataframe
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for images
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=2)
        
        # Setup left panel (dataframe)
        self.setup_dataframe_panel(left_frame)
        
        # Setup right panels (images and controls)
        self.setup_image_panels(right_paned)
        
        # Setup menu
        self.setup_menu()
        
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reload Data", command=self.load_data)
        file_menu.add_command(label="Save Changes", command=self.save_changes)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh Images", command=self.refresh_images)
        
    def setup_dataframe_panel(self, parent):
        # Title
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(title_frame, text="Sections Database", font=('Arial', 12, 'bold')).pack()
        
        # Dataframe will be added after data is loaded
        self.dataframe_container = ttk.Frame(parent)
        self.dataframe_container.pack(fill=tk.BOTH, expand=True)
        
    def setup_image_panels(self, parent):
        # Top panel: Section image and controls
        top_frame = ttk.Frame(parent)
        parent.add(top_frame, weight=1)
        
        # Section image frame
        section_frame = ttk.LabelFrame(top_frame, text="Section View")
        section_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.section_canvas = tk.Canvas(section_frame, bg='white', width=400, height=300)
        self.section_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(top_frame, text="Section Properties")
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Section info
        info_frame = ttk.Frame(controls_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.section_info_text = tk.Text(info_frame, height=8, width=30, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.section_info_text.yview)
        self.section_info_text.configure(yscrollcommand=info_scroll.set)
        self.section_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Toggle controls
        toggle_frame = ttk.Frame(controls_frame)
        toggle_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.mask_validated_var = tk.BooleanVar()
        self.include_image_var = tk.BooleanVar()
        
        ttk.Checkbutton(toggle_frame, text="Mask Validated", 
                       variable=self.mask_validated_var,
                       command=self.on_property_change).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(toggle_frame, text="Include Image", 
                       variable=self.include_image_var,
                       command=self.on_property_change).pack(anchor=tk.W, pady=2)
        
        # Downsampling control
        downsample_frame = ttk.Frame(controls_frame)
        downsample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(downsample_frame, text="Downsample Factor:").pack(anchor=tk.W)
        self.downsample_var = tk.IntVar(value=4)
        downsample_spinbox = ttk.Spinbox(downsample_frame, from_=1, to=20, 
                                        textvariable=self.downsample_var, width=10,
                                        command=self.on_downsample_change)
        downsample_spinbox.pack(anchor=tk.W, pady=2)
        downsample_spinbox.bind('<Return>', lambda e: self.on_downsample_change())
        
        # Action buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Open Section Folder", 
                  command=self.open_section_folder).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="View Full Size", 
                  command=self.view_full_size).pack(fill=tk.X, pady=2)
        
        # Bottom panel: Slide overview
        bottom_frame = ttk.LabelFrame(parent, text="Slide Overview")
        parent.add(bottom_frame, weight=1)
        
        self.slide_canvas = tk.Canvas(bottom_frame, bg='lightgray', width=800, height=300)
        self.slide_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_data(self):
        """Load all section data from tracking files"""
        try:
            data_dir = config['data_dir']
            tracking_dir = os.path.join(data_dir, 'tracking')
            
            if not os.path.exists(tracking_dir):
                messagebox.showerror("Error", f"Tracking directory not found: {tracking_dir}")
                return
            
            # Load tracking data
            suggested_regions_paths, basename_list, data_path_list = load_tracking_data(tracking_dir)
            
            if not suggested_regions_paths:
                messagebox.showwarning("Warning", "No tracking data found")
                return
            
            # Load suggested regions
            self.sections_df = load_suggested_regions(suggested_regions_paths, basename_list, data_path_list)
            
            # Add validation columns if they don't exist
            if 'mask_validated' not in self.sections_df.columns:
                self.sections_df['mask_validated'] = False
            if 'include_image' not in self.sections_df.columns:
                self.sections_df['include_image'] = True
            
            # Create filterable dataframe widget
            for widget in self.dataframe_container.winfo_children():
                widget.destroy()
                
            self.dataframe_widget = FilterableDataFrame(
                self.dataframe_container, 
                self.sections_df, 
                on_selection_change=self.on_section_select
            )
            
            messagebox.showinfo("Success", f"Loaded {len(self.sections_df)} sections")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def on_section_select(self, section_row):
        """Handle section selection from dataframe"""
        self.current_section = section_row
        self.update_section_info()
        self.load_section_image()
        self.load_slide_overview()
        
    def update_section_info(self):
        """Update the section information display"""
        if self.current_section is None:
            return
            
        info_text = f"Hash: {self.current_section['section_hash']}\n"
        info_text += f"Basename: {self.current_section['basename']}\n"
        info_text += f"Tissue Type: {self.current_section.get('tissue_type', 'N/A')}\n"
        info_text += f"Label Value: {self.current_section.get('label_values', 'N/A')}\n"
        info_text += f"Section Bounds: {self.current_section.get('section_bounds', 'N/A')}\n"
        info_text += f"Data Path: {self.current_section['data_path']}\n"
        
        self.section_info_text.delete(1.0, tk.END)
        self.section_info_text.insert(1.0, info_text)
        
        # Update toggle states
        self.mask_validated_var.set(bool(self.current_section.get('mask_validated', False)))
        self.include_image_var.set(bool(self.current_section.get('include_image', True)))
    
    def load_section_image(self):
        """Load and display the section image with prediction overlay"""
        if self.current_section is None:
            return
            
        try:
            # Get section image with downsampling
            section_hash = self.current_section['section_hash']
            downsample_factor = self.downsample_var.get()
            section, section_details = get_section_from_hash(section_hash, self.sections_df, down_sample=downsample_factor)
            
            self.section_image = section
            
            # Look for prediction mask
            mask_path = self.find_prediction_mask(section_hash)
            if mask_path and os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                # Downsample mask to match section image if needed
                if downsample_factor > 1:
                    new_height = mask.shape[0] // downsample_factor
                    new_width = mask.shape[1] // downsample_factor
                    self.section_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                else:
                    self.section_mask = mask
            else:
                self.section_mask = None
            
            self.display_section_image()
            
        except Exception as e:
            print(f"Error loading section image: {e}")
            self.section_canvas.delete("all")
            self.section_canvas.create_text(200, 150, text=f"Error loading image:\n{str(e)}", 
                                          fill="red", font=('Arial', 10))
    
    def find_prediction_mask(self, section_hash):
        """Find the prediction mask file for a given section hash"""
        data_dir = config['data_dir']
        suggested_regions_dir = os.path.join(data_dir, 'suggested_regions')
        
        # Search for mask files containing the hash
        mask_pattern = os.path.join(suggested_regions_dir, '**', 'masks', f'*{section_hash}_mask.png')
        mask_files = glob(mask_pattern, recursive=True)
        
        if mask_files:
            return mask_files[0]
        return None
    
    def display_section_image(self):
        """Display the section image with optional prediction overlay"""
        if self.section_image is None:
            return
            
        # Convert to PIL Image
        if isinstance(self.section_image, np.ndarray):
            if self.section_image.dtype != np.uint8:
                # Normalize to 0-255 range
                img_normalized = ((self.section_image - self.section_image.min()) / 
                                (self.section_image.max() - self.section_image.min()) * 255).astype(np.uint8)
            else:
                img_normalized = self.section_image
            
            pil_image = Image.fromarray(img_normalized)
        else:
            pil_image = self.section_image
        
        # Store original size before any resizing
        original_size = pil_image.size
        
        # Add prediction overlay if available (before resizing for display)
        if self.section_mask is not None:
            pil_image = self.add_prediction_overlay(pil_image, self.section_mask)
        
        # Resize to fit canvas
        canvas_width = self.section_canvas.winfo_width()
        canvas_height = self.section_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
            pil_image.thumbnail((canvas_width - 10, canvas_height - 10), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.section_photo = ImageTk.PhotoImage(pil_image)
        
        self.section_canvas.delete("all")
        canvas_center_x = canvas_width // 2 if canvas_width > 1 else 200
        canvas_center_y = canvas_height // 2 if canvas_height > 1 else 150
        
        self.section_canvas.create_image(canvas_center_x, canvas_center_y, 
                                       image=self.section_photo, anchor=tk.CENTER)
    
    def add_prediction_overlay(self, image, mask):
        """Add prediction contour overlay to the image"""
        # Convert PIL to numpy for processing
        img_array = np.array(image)
        
        # Ensure mask dimensions match image dimensions
        if mask.shape[:2] != img_array.shape[:2]:
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        
        # Create binary mask
        if mask_resized.max() > 1:
            binary_mask = (mask_resized > 127).astype(np.uint8)
        else:
            binary_mask = mask_resized.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on image
        if len(contours) > 0:
            cv2.drawContours(img_array, contours, -1, (255, 255, 0), 2)  # Yellow contours
        
        return Image.fromarray(img_array)
    
    def load_slide_overview(self):
        """Load and display the slide overview with section highlights"""
        if self.current_section is None:
            return
            
        try:
            slide_path = self.current_section['data_path']
            
            # Load slide if different from current
            if slide_path != self.current_slide_path:
                self.current_slide_path = slide_path
                slide = slideio.open_slide(slide_path, 'SVS')
                self.current_slide_scene = slide.get_scene(0)
            
            # Get all sections for this slide
            slide_sections = self.sections_df[self.sections_df['data_path'] == slide_path]
            
            # Create visualization
            self.display_slide_overview(slide_sections)
            
        except Exception as e:
            print(f"Error loading slide overview: {e}")
            self.slide_canvas.delete("all")
            self.slide_canvas.create_text(400, 150, text=f"Error loading slide:\n{str(e)}", 
                                        fill="red", font=('Arial', 10))
    
    def display_slide_overview(self, slide_sections):
        """Display slide overview with section highlights"""
        if self.current_slide_scene is None:
            return
            
        try:
            # Get slide image at low resolution
            slide_rect = self.current_slide_scene.rect
            down_sample = max(slide_rect[2] // 800, slide_rect[3] // 600, 1)
            
            slide_image = self.current_slide_scene.read_block(
                size=(slide_rect[2] // down_sample, slide_rect[3] // down_sample)
            )
            
            # Convert to PIL
            if isinstance(slide_image, np.ndarray):
                pil_slide = Image.fromarray(slide_image)
            else:
                pil_slide = slide_image
            
            # Draw section rectangles
            draw = ImageDraw.Draw(pil_slide)
            
            current_hash = self.current_section['section_hash']
            
            for _, section in slide_sections.iterrows():
                try:
                    bounds = literal_eval(section['section_bounds'])
                    # Scale bounds to match downsampled image
                    scaled_bounds = [b // down_sample for b in bounds]
                    
                    # bounds format is [row_min, col_min, row_max, col_max]
                    # PIL rectangle expects [x1, y1, x2, y2] where x=col, y=row
                    x1, y1, x2, y2 = scaled_bounds[1], scaled_bounds[0], scaled_bounds[3], scaled_bounds[2]
                    
                    # Choose color and style based on whether this is the current section
                    if section['section_hash'] == current_hash:
                        # Fill current section with red and add red outline
                        draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 128), outline=(255, 0, 0), width=3)
                    else:
                        # Just outline for other sections
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                    
                except Exception as e:
                    print(f"Error drawing section {section['section_hash']}: {e}")
                    continue
            
            # Resize to fit canvas
            canvas_width = self.slide_canvas.winfo_width()
            canvas_height = self.slide_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                pil_slide.thumbnail((canvas_width - 10, canvas_height - 10), Image.Resampling.LANCZOS)
            
            # Display
            self.slide_photo = ImageTk.PhotoImage(pil_slide)
            
            self.slide_canvas.delete("all")
            canvas_center_x = canvas_width // 2 if canvas_width > 1 else 400
            canvas_center_y = canvas_height // 2 if canvas_height > 1 else 150
            
            self.slide_canvas.create_image(canvas_center_x, canvas_center_y, 
                                         image=self.slide_photo, anchor=tk.CENTER)
            
        except Exception as e:
            print(f"Error displaying slide overview: {e}")
    
    def on_property_change(self):
        """Handle changes to section properties"""
        if self.current_section is None:
            return
            
        # Update the dataframe
        section_hash = self.current_section['section_hash']
        mask = self.sections_df['section_hash'] == section_hash
        
        self.sections_df.loc[mask, 'mask_validated'] = self.mask_validated_var.get()
        self.sections_df.loc[mask, 'include_image'] = self.include_image_var.get()
        
        # Update the current section reference
        self.current_section = self.sections_df[mask].iloc[0]
        
        # Update the dataframe widget
        self.dataframe_widget.update_data(self.sections_df)
        
        # Immediately write changes back to the section frame CSV
        self.save_section_changes(self.current_section)
    
    def save_section_changes(self, section):
        """Save changes for a specific section back to its CSV file"""
        try:
            basename = section['basename']
            
            # Find the corresponding CSV file
            csv_files = glob(os.path.join(config['data_dir'], 'suggested_regions', 
                                        f"*{basename}*", f"*{basename}*_section_frame.csv"))
            
            if csv_files:
                csv_path = csv_files[0]
                
                # Load the original CSV
                original_df = pd.read_csv(csv_path)
                
                # Get all sections for this basename from our current dataframe
                basename_sections = self.sections_df[self.sections_df['basename'] == basename]
                
                # Update the original dataframe with our changes
                for _, updated_section in basename_sections.iterrows():
                    section_hash = updated_section['section_hash']
                    mask = original_df['section_hash'] == section_hash
                    
                    if mask.any():
                        # Update existing columns
                        for col in ['mask_validated', 'include_image']:
                            if col in updated_section:
                                if col not in original_df.columns:
                                    original_df[col] = False  # Initialize column if it doesn't exist
                                original_df.loc[mask, col] = updated_section[col]
                
                # Save back to CSV
                original_df.to_csv(csv_path, index=False)
                
        except Exception as e:
            print(f"Error saving section changes: {e}")
    
    def save_changes(self):
        """Save changes back to the CSV files"""
        try:
            # Group by basename and save each group to its corresponding CSV
            for basename, group in self.sections_df.groupby('basename'):
                # Find the corresponding CSV file
                csv_files = glob(os.path.join(config['data_dir'], 'suggested_regions', 
                                            f"*{basename}*", f"*{basename}*_section_frame.csv"))
                
                if csv_files:
                    csv_path = csv_files[0]
                    # Save only the columns that were in the original file
                    original_df = pd.read_csv(csv_path)
                    columns_to_save = list(original_df.columns) + ['mask_validated', 'include_image']
                    
                    # Ensure we only save columns that exist
                    available_columns = [col for col in columns_to_save if col in group.columns]
                    group[available_columns].to_csv(csv_path, index=False)
            
            messagebox.showinfo("Success", "Changes saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save changes: {str(e)}")
    
    def on_downsample_change(self):
        """Handle changes to downsample factor"""
        if self.current_section is not None:
            self.load_section_image()
    
    def refresh_images(self):
        """Refresh the current images"""
        if self.current_section is not None:
            self.load_section_image()
            self.load_slide_overview()
    
    def open_section_folder(self):
        """Open the folder containing the current section"""
        if self.current_section is None:
            return
            
        try:
            section_hash = self.current_section['section_hash']
            basename = self.current_section['basename']
            
            # Find the section folder
            suggested_regions_dir = os.path.join(config['data_dir'], 'suggested_regions')
            section_folders = glob(os.path.join(suggested_regions_dir, f"*{basename}*"))
            
            if section_folders:
                folder_path = section_folders[0]
                # Open folder in file manager
                if os.name == 'nt':  # Windows
                    os.startfile(folder_path)
                elif os.name == 'posix':  # macOS and Linux
                    os.system(f'xdg-open "{folder_path}"')
            else:
                messagebox.showwarning("Warning", "Section folder not found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
    
    def view_full_size(self):
        """Open a new window with full-size section image"""
        if self.section_image is None:
            return
            
        # Create new window
        full_size_window = tk.Toplevel(self.root)
        full_size_window.title(f"Full Size - {self.current_section['section_hash']}")
        full_size_window.geometry("800x600")
        
        # Create canvas with scrollbars
        canvas_frame = ttk.Frame(full_size_window)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Load full-size image (without downsampling)
        try:
            section_hash = self.current_section['section_hash']
            full_section, _ = get_section_from_hash(section_hash, self.sections_df, down_sample=1)
            
            if isinstance(full_section, np.ndarray):
                if full_section.dtype != np.uint8:
                    img_normalized = ((full_section - full_section.min()) / 
                                    (full_section.max() - full_section.min()) * 255).astype(np.uint8)
                else:
                    img_normalized = full_section
                pil_image = Image.fromarray(img_normalized)
            else:
                pil_image = full_section
            
            # Load full-size mask if available
            mask_path = self.find_prediction_mask(section_hash)
            if mask_path and os.path.exists(mask_path):
                full_mask = np.array(Image.open(mask_path))
                pil_image = self.add_prediction_overlay(pil_image, full_mask)
                
        except Exception as e:
            print(f"Error loading full-size image: {e}")
            return
        
        photo = ImageTk.PhotoImage(pil_image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Keep reference to prevent garbage collection
        canvas.image = photo
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main function to run the section viewer GUI"""
    app = SectionViewer()
    app.run()


if __name__ == "__main__":
    main()
