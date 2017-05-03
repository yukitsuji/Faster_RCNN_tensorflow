#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import rospy
import numpy as np
import cv2
import pcl
import glob
import math
from parse_xml import parseXML
from util import *
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

def get_ALL_Image(image_dir, label_dir):
    """Get Images and ROIs of All Datasets.
    # Args:
        image_dir  (str): path of image directory.
        label_dir    (str): path of label's xml directory.
        num_of_rois(int): Number of ROIs in a image.
    # Returns:
        images     (list): ndarray Images of datasets.
        pred_bboxes(ndarray): rescaled bbox Label [0, x, y, w, h]
    """
    # 車が含まれている画像のみラベルと一緒に読み込む
    image_pathlist = 0 #load_for_detection(label_dir)
    dataset_img_list = [] # len(dataset_img_list) == Number of Datasets Images
    # Preprocess Ground Truth ROIs. shape is [Num of ROIs * batch_size, x, y, w, h, 0, 1]
    g_bboxes = []
    # shape is [batch_channel, x, y, w, h]
    image_pathlist = glob.glob(image_dir)
    label_pathlist = glob.glob(label_dir)
    image_pathlist.sort()
    label_pathlist.sort()

    for index, (image_path, label_path) in enumerate(zip(image_pathlist, label_pathlist)):
        if index == 100:
            break
        img = cv2.imread(image_path)
        label = read_label_from_txt(label_path)
        if label is None:
            continue
        img, im_scale = preprocess_imgs(img)
        dataset_img_list.append(img)
        g_bboxes.append(label)

    g_bboxes = np.array(g_bboxes)
    return np.array(dataset_img_list), g_bboxes
