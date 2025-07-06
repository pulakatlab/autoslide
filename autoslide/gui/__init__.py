"""
GUI components for AutoSlide
"""

from .section_viewer import SectionViewer
from .annotation_gui import AnnotationGUI

def main_section_viewer():
    """Launch the section viewer GUI"""
    from .section_viewer import main
    main()

def main_annotation_gui():
    """Launch the annotation GUI"""
    from .annotation_gui import main
    main()

__all__ = ['SectionViewer', 'AnnotationGUI', 'main_section_viewer', 'main_annotation_gui']
