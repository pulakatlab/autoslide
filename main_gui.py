import tkinter as tk
from tkinter import messagebox
import subprocess

def run_annotation_gui():
    subprocess.run(["python", "path/to/annotation_gui.py"])

def run_suggest_regions():
    subprocess.run(["python", "autoslide/pipeline/suggest_regions.py"])

def run_model_predictions():
    subprocess.run(["python", "autoslide/pipeline/model/prediction.py"])

def run_section_viewer():
    subprocess.run(["python", "path/to/section_viewer.py"])

root = tk.Tk()
root.title("Main GUI")

tk.Button(root, text="Run Annotation GUI", command=run_annotation_gui).pack()
tk.Button(root, text="Run Suggest Regions", command=run_suggest_regions).pack()
tk.Button(root, text="Run Model Predictions", command=run_model_predictions).pack()
tk.Button(root, text="Run Section Viewer", command=run_section_viewer).pack()

root.mainloop()
