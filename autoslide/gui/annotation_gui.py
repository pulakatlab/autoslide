"""
GUI for initial and final annotation of slide sections.

This module provides a comprehensive interface for:
1. Running initial annotation to detect tissue regions
2. Editing tissue annotations in a dataframe
3. Finalizing annotations with tissue type assignments
4. Visualizing results throughout the process
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json
from glob import glob
from tqdm import tqdm
import slideio
import cv2 as cv
from skimage.morphology import binary_dilation as dilation
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters.rank import gradient
from scipy.ndimage import binary_fill_holes
from ast import literal_eval

from autoslide import config
from autoslide.pipeline.utils import get_threshold_mask


class EditableDataFrame:
    """A widget for editing dataframe values with validation"""
    
    def __init__(self, parent, data, editable_columns=None, on_change=None):
        self.parent = parent
        self.data = data.copy()
        self.editable_columns = editable_columns or []
        self.on_change = on_change
        
        self.setup_ui()
        self.populate_tree()
        
    def setup_ui(self):
        # Create main frame
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview frame with scrollbars
        tree_frame = ttk.Frame(self.frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview
        columns = list(self.data.columns)
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col)
            width = 150 if col in self.editable_columns else 100
            self.tree.column(col, width=width, minwidth=50)
        
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
        
        # Bind double-click for editing
        self.tree.bind('<Double-1>', self.on_double_click)
        
        # Edit controls frame
        edit_frame = ttk.Frame(self.frame)
        edit_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(edit_frame, text="Edit selected cell:").pack(side=tk.LEFT)
        self.edit_var = tk.StringVar()
        self.edit_entry = ttk.Entry(edit_frame, textvariable=self.edit_var, width=20)
        self.edit_entry.pack(side=tk.LEFT, padx=5)
        self.edit_entry.bind('<Return>', self.apply_edit)
        
        ttk.Button(edit_frame, text="Apply", command=self.apply_edit).pack(side=tk.LEFT, padx=5)
        
        # Tissue type presets for quick entry
        ttk.Label(edit_frame, text="Quick tissue types:").pack(side=tk.LEFT, padx=(20, 5))
        for tissue_type in ['heart', 'liver', 'kidney', 'lung', 'muscle']:
            ttk.Button(edit_frame, text=tissue_type, 
                      command=lambda t=tissue_type: self.quick_tissue_type(t)).pack(side=tk.LEFT, padx=2)
        
    def populate_tree(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add data
        for idx, row in self.data.iterrows():
            values = []
            for col in self.data.columns:
                val = row[col]
                if pd.isna(val):
                    values.append("")
                else:
                    values.append(str(val))
            self.tree.insert('', tk.END, values=values, tags=(idx,))
    
    def on_double_click(self, event):
        """Handle double-click to edit cell"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if not item:
            return
            
        # Get column clicked
        column = self.tree.identify_column(event.x)
        if not column:
            return
            
        # Convert column identifier to column name
        col_index = int(column.replace('#', '')) - 1
        if col_index < 0 or col_index >= len(self.data.columns):
            return
            
        col_name = list(self.data.columns)[col_index]
        
        # Only allow editing of specified columns
        if col_name not in self.editable_columns:
            return
            
        # Get current value
        item_data = self.tree.item(item)
        current_value = item_data['values'][col_index]
        
        # Set up for editing
        self.edit_var.set(str(current_value))
        self.edit_entry.focus()
        self.current_edit_item = item
        self.current_edit_column = col_name
        
    def apply_edit(self, event=None):
        """Apply the current edit"""
        if not hasattr(self, 'current_edit_item') or not self.current_edit_item:
            return
            
        new_value = self.edit_var.get()
        
        # Get row index
        item_data = self.tree.item(self.current_edit_item)
        row_idx = int(item_data['tags'][0])
        
        # Validate and convert value based on column type
        try:
            if self.current_edit_column == 'tissue_num':
                if new_value.strip():
                    new_value = int(new_value)
                else:
                    new_value = np.nan
            elif self.current_edit_column == 'tissue_type':
                new_value = str(new_value).strip() if new_value.strip() else np.nan
                
            # Update data
            self.data.loc[row_idx, self.current_edit_column] = new_value
            
            # Refresh display
            self.populate_tree()
            
            # Clear edit state
            self.edit_var.set("")
            self.current_edit_item = None
            self.current_edit_column = None
            
            # Notify of change
            if self.on_change:
                self.on_change()
                
        except ValueError as e:
            messagebox.showerror("Invalid Value", f"Invalid value for {self.current_edit_column}: {e}")
    
    def quick_tissue_type(self, tissue_type):
        """Quickly set tissue type for selected row"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a row first")
            return
            
        item = selection[0]
        item_data = self.tree.item(item)
        row_idx = int(item_data['tags'][0])
        
        # Set tissue type
        self.data.loc[row_idx, 'tissue_type'] = tissue_type
        
        # Auto-assign tissue_num if not set
        if pd.isna(self.data.loc[row_idx, 'tissue_num']):
            # Find next available tissue number
            existing_nums = self.data['tissue_num'].dropna().astype(int).tolist()
            next_num = 1
            while next_num in existing_nums:
                next_num += 1
            self.data.loc[row_idx, 'tissue_num'] = next_num
        
        self.populate_tree()
        
        if self.on_change:
            self.on_change()
    
    def get_data(self):
        """Get the current dataframe"""
        return self.data.copy()


class AnnotationGUI:
    """Main GUI application for slide annotation"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AutoSlide Annotation Tool")
        self.root.geometry("1600x1000")
        
        # Data
        self.slide_files = []
        self.current_slide_path = None
        self.current_slide = None
        self.current_scene = None
        self.current_metadata = None
        self.current_mask = None
        self.current_image = None
        
        # Annotation parameters
        self.down_sample = 100
        self.dilation_kern_size = 2
        self.area_threshold = 10000
        
        self.setup_ui()
        self.check_and_process_initial_annotations()
        
    def setup_ui(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Annotation Editing
        self.setup_editing_tab()
        
        # Setup menu
        self.setup_menu()
        
        # Setup status bar
        self.setup_status_bar()
        
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Process Missing Initial Annotations", command=self.process_missing_annotations)
        file_menu.add_command(label="Open Data Directory", command=self.open_data_directory)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Annotation Parameters", command=self.show_parameters_dialog)
    
    def setup_status_bar(self):
        """Setup the status bar at the bottom of the window"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        
        
    def setup_editing_tab(self):
        """Setup the annotation editing tab"""
        editing_frame = ttk.Frame(self.notebook)
        self.notebook.add(editing_frame, text="Edit Annotations")
        
        # Top frame for file selection
        edit_file_frame = ttk.LabelFrame(editing_frame, text="Select Processed File")
        edit_file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File list frame with scrollable listbox
        file_list_frame = ttk.Frame(edit_file_frame)
        file_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Listbox with scrollbar
        list_container = ttk.Frame(file_list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.processed_files_listbox = tk.Listbox(list_container, height=8, selectmode=tk.SINGLE)
        files_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.processed_files_listbox.yview)
        self.processed_files_listbox.configure(yscrollcommand=files_scrollbar.set)
        
        self.processed_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to load file
        self.processed_files_listbox.bind('<Double-Button-1>', lambda e: self.load_for_editing())
        
        # Control buttons
        edit_controls = ttk.Frame(edit_file_frame)
        edit_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(edit_controls, text="Load for Editing", 
                  command=self.load_for_editing).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_controls, text="Refresh List", 
                  command=self.refresh_processed_files).pack(side=tk.LEFT, padx=5)
        
        # Main editing area - split between dataframe and image
        edit_paned = ttk.PanedWindow(editing_frame, orient=tk.HORIZONTAL)
        edit_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Dataframe editing
        df_frame = ttk.LabelFrame(edit_paned, text="Region Annotations")
        edit_paned.add(df_frame, weight=1)
        
        # Placeholder for dataframe widget
        self.dataframe_container = ttk.Frame(df_frame)
        self.dataframe_container.pack(fill=tk.BOTH, expand=True)
        
        # Right: Image display
        img_frame = ttk.LabelFrame(edit_paned, text="Slide Preview")
        edit_paned.add(img_frame, weight=1)
        
        self.edit_fig, self.edit_ax = plt.subplots(1, 1, figsize=(8, 8))
        self.edit_canvas = FigureCanvasTkAgg(self.edit_fig, img_frame)
        self.edit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.edit_ax.axis('off')
        
        
    def check_and_process_initial_annotations(self):
        """Check for missing initial annotations and process them automatically"""
        data_dir = config['data_dir']
        glob_pattern = 'TRI*.svs'
        slide_files = glob(os.path.join(data_dir, '**', glob_pattern), recursive=True)
        
        annot_dir = os.path.join(data_dir, 'initial_annotation')
        tracking_dir = os.path.join(data_dir, 'tracking')
        
        missing_files = []
        for file_path in slide_files:
            file_basename = os.path.basename(file_path)
            csv_path = os.path.join(annot_dir, file_basename.replace('.svs', '.csv'))
            json_path = os.path.join(tracking_dir, file_basename.replace('.svs', '.json'))
            
            if not os.path.exists(csv_path) or not os.path.exists(json_path):
                missing_files.append(file_path)
        
        if missing_files:
            response = messagebox.askyesno(
                "Missing Initial Annotations",
                f"Found {len(missing_files)} files without initial annotations. Process them now?"
            )
            if response:
                self.process_files_automatically(missing_files)
        
        # Refresh processed files list
        self.refresh_processed_files()
    
    def process_missing_annotations(self):
        """Menu command to process missing annotations"""
        self.check_and_process_initial_annotations()
    
    def process_files_automatically(self, file_list):
        """Process a list of files automatically"""
        total_files = len(file_list)
        
        for i, file_path in enumerate(file_list):
            try:
                self.process_single_file(file_path, show_preview=False)
                # Update progress if we had a progress bar
                progress = ((i + 1) / total_files) * 100
                self.root.update()
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        messagebox.showinfo("Complete", f"Processed {len(file_list)} files")
        
    def refresh_processed_files(self):
        """Refresh the list of processed files"""
        data_dir = config['data_dir']
        annot_dir = os.path.join(data_dir, 'initial_annotation')
        
        # Clear current list
        self.processed_files_listbox.delete(0, tk.END)
        
        if os.path.exists(annot_dir):
            csv_files = glob(os.path.join(annot_dir, '*.csv'))
            basenames = [os.path.basename(f).replace('.csv', '') for f in csv_files]
            
            # Add files to listbox
            for basename in sorted(basenames):
                self.processed_files_listbox.insert(tk.END, basename)
            
            # Select first item if available
            if basenames:
                self.processed_files_listbox.selection_set(0)
        
        
    def process_single_file(self, file_path, show_preview=True):
        """Process a single slide file for initial annotation"""
        try:
            self.status_var.set(f"Processing {os.path.basename(file_path)}...")
            self.root.update()
            
            # Setup directories
            data_dir = config['data_dir']
            annot_dir = os.path.join(data_dir, 'initial_annotation')
            tracking_dir = os.path.join(data_dir, 'tracking')
            os.makedirs(annot_dir, exist_ok=True)
            os.makedirs(tracking_dir, exist_ok=True)
            
            file_basename = os.path.basename(file_path)
            
            # Load slide
            slide = slideio.open_slide(file_path, 'SVS')
            scene = slide.get_scene(0)
            image_rect = np.array(scene.rect) // self.down_sample
            image = scene.read_block(size=image_rect[2:])
            
            # Get threshold mask
            threshold_mask = get_threshold_mask(scene, down_sample=self.down_sample)
            
            # Dilate mask
            dilation_kern = np.ones((self.dilation_kern_size, self.dilation_kern_size))
            dilated_mask = dilation(threshold_mask, footprint=dilation_kern)
            
            # Label regions
            label_image = label(dilated_mask)
            regions = regionprops(label_image)
            
            # Extract region features
            wanted_feature_names = [
                'label', 'area', 'eccentricity', 'axis_major_length',
                'axis_minor_length', 'solidity', 'centroid',
            ]
            
            wanted_features = [
                [getattr(region, feature_name) for region in regions]
                for feature_name in wanted_feature_names
            ]
            
            region_frame = pd.DataFrame(
                {feature_name: feature for feature_name, feature in
                 zip(wanted_feature_names, wanted_features)}
            )
            
            region_frame.sort_values('label', ascending=True, inplace=True)
            wanted_regions_frame = region_frame[region_frame.area > self.area_threshold]
            wanted_regions = wanted_regions_frame.label.values
            
            # Add annotation columns
            wanted_regions_frame['tissue_type'] = np.nan
            wanted_regions_frame['tissue_num'] = np.nan
            
            # Save CSV
            csv_path = os.path.join(annot_dir, file_basename.replace('.svs', '.csv'))
            wanted_regions_frame.to_csv(csv_path, index=False)
            
            # Create final label image
            fin_label_image = label_image.copy()
            for i in region_frame.label.values:
                if i not in wanted_regions:
                    fin_label_image[fin_label_image == i] = 0
            
            # Save mask
            mask_path = os.path.join(annot_dir, file_basename.replace('.svs', '.npy'))
            np.save(mask_path, fin_label_image)
            
            # Create tracking JSON
            json_data = {
                'file_basename': file_basename,
                'data_path': file_path,
                'initial_mask_path': mask_path,
                'wanted_regions_frame_path': csv_path,
            }
            
            json_path = os.path.join(tracking_dir, file_basename.replace('.svs', '.json'))
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            print(f"Completed processing {file_basename}")
            
        except Exception as e:
            self.status_var.set(f"Error processing {os.path.basename(file_path)}")
            raise e
            
        
    def load_for_editing(self):
        """Load a processed file for editing"""
        selection = self.processed_files_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a processed file")
            return
            
        selected_file = self.processed_files_listbox.get(selection[0])
            
        try:
            data_dir = config['data_dir']
            annot_dir = os.path.join(data_dir, 'initial_annotation')
            
            # Load CSV
            csv_path = os.path.join(annot_dir, selected_file + '.csv')
            if not os.path.exists(csv_path):
                messagebox.showerror("File Not Found", f"CSV file not found: {csv_path}")
                return
                
            self.current_metadata = pd.read_csv(csv_path)
            
            # Load mask
            mask_path = os.path.join(annot_dir, selected_file + '.npy')
            if os.path.exists(mask_path):
                self.current_mask = np.load(mask_path)
            
            # Create editable dataframe
            for widget in self.dataframe_container.winfo_children():
                widget.destroy()
                
            self.dataframe_widget = EditableDataFrame(
                self.dataframe_container,
                self.current_metadata,
                editable_columns=['tissue_type', 'tissue_num'],
                on_change=self.on_annotation_change
            )
            
            # Update image preview
            self.update_edit_preview()
            
            self.status_var.set(f"Loaded {selected_file} for editing")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            
    def on_annotation_change(self):
        """Handle changes to annotations"""
        self.current_metadata = self.dataframe_widget.get_data()
        self.update_edit_preview()
        # Automatically save changes
        self.save_current_annotations()
        # Automatically finalize annotation
        self.auto_finalize_annotation()
        
    def update_edit_preview(self):
        """Update the editing preview"""
        if self.current_mask is None:
            return
            
        self.edit_ax.clear()
        
        # Create label overlay
        image_label_overlay = label2rgb(self.current_mask, 
                                      image=self.current_mask > 0, 
                                      bg_label=0)
        self.edit_ax.imshow(image_label_overlay)
        
        # Add annotations
        for _, row in self.current_metadata.iterrows():
            if not pd.isna(row['tissue_type']):
                centroid = literal_eval(str(row['centroid']))
                tissue_str = f"{int(row['tissue_num']) if not pd.isna(row['tissue_num']) else '?'}_{row['tissue_type']}"
                self.edit_ax.text(centroid[1], centroid[0], tissue_str,
                                color='red', fontsize=10, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        self.edit_ax.set_title('Annotated Regions')
        self.edit_ax.axis('off')
        self.edit_canvas.draw()
        
    def save_current_annotations(self):
        """Save current annotations to CSV"""
        if self.current_metadata is None:
            return
            
        selection = self.processed_files_listbox.curselection()
        if not selection:
            return
            
        selected_file = self.processed_files_listbox.get(selection[0])
            
        data_dir = config['data_dir']
        annot_dir = os.path.join(data_dir, 'initial_annotation')
        csv_path = os.path.join(annot_dir, selected_file + '.csv')
        
        self.current_metadata.to_csv(csv_path, index=False)
        
    def auto_finalize_annotation(self):
        """Automatically finalize the current annotation"""
        selection = self.processed_files_listbox.curselection()
        if not selection:
            return
            
        selected_file = self.processed_files_listbox.get(selection[0])
            
        if self.current_metadata is None or self.current_mask is None:
            return
            
        try:
            data_dir = config['data_dir']
            fin_annotation_dir = os.path.join(data_dir, 'final_annotation')
            tracking_dir = os.path.join(data_dir, 'tracking')
            os.makedirs(fin_annotation_dir, exist_ok=True)
            
            # Create final mask
            final_mask = self.current_mask.copy()
            
            # Apply label mapping
            label_map = {}
            for _, row in self.current_metadata.iterrows():
                if not pd.isna(row['tissue_num']):
                    label_map[int(row['label'])] = int(row['tissue_num'])
            
            for old_label, new_label in label_map.items():
                final_mask[final_mask == old_label] = new_label
            
            # Save final mask
            final_mask_path = os.path.join(fin_annotation_dir, selected_file + '.npy')
            np.save(final_mask_path, final_mask)
            
            # Create final visualization
            image_label_overlay = label2rgb(final_mask, image=final_mask > 0, bg_label=0)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image_label_overlay)
            ax.set_title(selected_file)
            
            # Add tissue labels
            for _, row in self.current_metadata.iterrows():
                if not pd.isna(row['tissue_type']) and not pd.isna(row['tissue_num']):
                    centroid = literal_eval(str(row['centroid']))
                    tissue_str = f"{int(row['tissue_num'])}_{row['tissue_type']}"
                    ax.text(centroid[1], centroid[0], tissue_str,
                           color='red', fontsize=12, weight='bold')
            
            ax.axis('off')
            
            # Save visualization
            vis_path = os.path.join(fin_annotation_dir, selected_file + '.png')
            fig.savefig(vis_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # Update tracking JSON
            json_path = os.path.join(tracking_dir, selected_file + '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                json_data['fin_mask_path'] = final_mask_path
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
            
        except Exception as e:
            print(f"Error auto-finalizing annotation: {str(e)}")
        
            
    def show_parameters_dialog(self):
        """Show dialog for editing annotation parameters"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Annotation Parameters")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Parameters frame
        params_frame = ttk.LabelFrame(dialog, text="Processing Parameters")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Down sample
        ttk.Label(params_frame, text="Down Sample Factor:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        down_sample_var = tk.IntVar(value=self.down_sample)
        ttk.Spinbox(params_frame, from_=10, to=500, textvariable=down_sample_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Dilation kernel size
        ttk.Label(params_frame, text="Dilation Kernel Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        dilation_var = tk.IntVar(value=self.dilation_kern_size)
        ttk.Spinbox(params_frame, from_=1, to=10, textvariable=dilation_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Area threshold
        ttk.Label(params_frame, text="Area Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        area_var = tk.IntVar(value=self.area_threshold)
        ttk.Spinbox(params_frame, from_=1000, to=100000, textvariable=area_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def apply_params():
            self.down_sample = down_sample_var.get()
            self.dilation_kern_size = dilation_var.get()
            self.area_threshold = area_var.get()
            dialog.destroy()
            
        ttk.Button(button_frame, text="Apply", command=apply_params).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        
    def open_data_directory(self):
        """Open the data directory in file manager"""
        data_dir = config['data_dir']
        if os.name == 'nt':  # Windows
            os.startfile(data_dir)
        elif os.name == 'posix':  # macOS and Linux
            os.system(f'xdg-open "{data_dir}"')
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main function to run the annotation GUI"""
    app = AnnotationGUI()
    app.run()


if __name__ == "__main__":
    main()
