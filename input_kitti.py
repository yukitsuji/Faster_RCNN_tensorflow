#!/usr/bin/env python
import sys
import os
import rospy
import numpy as np
import cv2
import pcl
import glob
import math
from parse_xml import parseXML
import matplotlib.pyplot as plt

def read_label_from_txt(label_path):
    """From label text file, Read bounding box
    Each text file corresponds to one image.

    # Returns:
        bounding_box(list): List of Bounding Boxes in one image
    """
    bounding_box = []
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            label = label.split(" ")
            if label[0] == ("Car" or "Van"): #  or "Truck"
                bounding_box.append(label[4:8])

    if bounding_box:
        return np.array(bounding_box, dtype=np.float64)
    else:
        return None
