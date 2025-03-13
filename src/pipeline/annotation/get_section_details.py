"""
Given a section id, this script will return the details of the section from 
the original data file.
"""

auto_slide_dir = '/media/bigdata/projects/pulakat_lab/auto_slide'
# auto_slide_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'

import os
import sys
sys.path.append(os.path.join(auto_slide_dir, 'src', 'pipeline'))
import utils

import slideio
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np 
from pprint import pprint
import pandas as pd
from skimage import morphology as morph
from scipy.ndimage import binary_fill_holes
from glob import glob

