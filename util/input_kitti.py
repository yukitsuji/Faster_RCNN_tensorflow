#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import cv2
import glob
import math
from parse_xml import parseXML
from data_util import *
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

def select_inputs_from_datasets(dataset_img_list, g_boxes, batch_size):
    """
    # Args:
        dataset_img_list      (ndarray): ndarray Images in datasets.
        g_boxes              (ndarray): GroundTruth Bounding Box with Class Label
                                         shape is [batch, 6*max_label_num]
                                         label is [x, y, w, h]
        batch_size                (int): batch size for training
    # Returns:
        batch_imgs    (ndarray): input batch images for Network. Shape is [Batch Size, shape]
        batch_g_boxes(ndarray): input GroundTruth Bounding Box for Network.
                                 Shape is [Num of ROIs*Batch Size]
    """
    perm = np.random.permutation(len(dataset_img_list))
    batches = [perm[i * batch_size:(i + 1) * batch_size] \
                   for i in range(len(dataset_img_list) // batch_size)]
    for batch in batches:
        batch_imgs = dataset_img_list[batch]
        batch_g_boxes = g_bboxes[batch]
        # Flip Conversion
        # batch_imgs, batch_p_bboxes, batch_g_boxes = flip_conversion(batch_imgs, batch_p_bboxes, batch_g_boxes)
        batch_imgs, batch_g_boxes = convert_imgslist_to_ndarray(batch_imgs, batch_g_boxes)
        yield batch_imgs, batch_g_boxes

# def convert_imgslist_to_ndarray(images, batch_g_boxes):
#     """Convert a list of images into a network input.
#     Assumes images are already prepared (means subtracted, BGR order, ...).
#
#     In this stage, the shape of images are different
#     """
#     max_shape = np.array([im.shape for im in images]).max(axis=0)
#     num_images = len(images)
#     blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
#                     dtype=np.float32)
#     for i in xrange(num_images):
#         if np.random.randint(2):
#             blob[i, 0:im.shape[0], 0:im.shape[1], :] = images[i][:, ::-1]
#             batch_g_boxes[i][:, 0] -= (max_shape[1]-1)
#             batch_g_boxes[i][:, 1] -= (max_shape[1]-1)
#             batch_g_boxes[i][:, 2] -= (max_shape[1]-1)
#             batch_g_boxes[i][:, 3] -= (max_shape[1]-1)
#         else:
#             blob[i, 0:im.shape[0], 0:im.shape[1], :] = images[i]
#     return blob, batch_g_boxes

def get_pathlist(image_dir, label_dir):
    image_pathlist = 0 #load_for_detection(label_dir)
    dataset_img_list = [] # len(dataset_img_list) == Number of Datasets Images
    # Preprocess Ground Truth ROIs. shape is [Num of ROIs * batch_size, x, y, w, h, 0, 1]
    g_bboxes = []
    # shape is [batch_channel, x, y, w, h]
    image_pathlist = glob.glob(image_dir)
    label_pathlist = glob.glob(label_dir)
    image_pathlist.sort()
    label_pathlist.sort()
    return np.array(image_pathlist), np.array(label_pathlist)

def generator__Image_and_label(image_pathlist, label_pathlist, batch_size=32):
    """Get Images and ROIs of All Datasets.
    # Args:
        image_pathlist  (ndarray): path of image files.
        label_pathlist    (ndarray): path of label's xml files.
        batch_size(int): Batch Size for network input.
    # Returns:
        images     (list): ndarray Images of datasets.
        g_bboxes(ndarray): rescaled bbox Label. Shapeis [Batch, ?, 4](x, y, w, h)
    """
    iter_num = image_pathlist.shape[0] / batch_size
    for it in range(iter_num):
        dataset_img_list = [] # len(dataset_img_list) == Number of Datasets Images
        g_bboxes = []
        for (image_path, label_path) in zip(image_pathlist[it*batch_size:(it+1)*batch_size], label_pathlist[it*batch_size:(it+1)*batch_size]):
            img = cv2.imread(image_path)
            label = read_label_from_txt(label_path)
            if label is None:
                continue
            img, im_scale = preprocess_imgs(img)
            dataset_img_list.append(img)
            g_bboxes.append(label)
        dataset_img_list = convert_imgslist_to_ndarray(dataset_img_list)
        # print img.shape
        # print im_scale
        # g_bboxes = unique_bboxes(g_bboxes, im_scale, feature_scale=1./16)
        print dataset_img_list.shape
        yield np.array(dataset_img_list), np.array(g_bboxes)

def get_ALL_Image(image_dir, label_dir):
    """Get Images and ROIs of All Datasets.
    # Args:
        image_dir  (str): path of image directory.
        label_dir    (str): path of label's xml directory.
        num_of_rois(int): Number of ROIs in a image.
    # Returns:
        images     (list): ndarray Images of datasets.
        pred_bboxes(ndarray): rescaled bbox Label. Shapeis [Batch, ?, 4](x, y, w, h)
    """
    import time
    start = time.time()
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

    for (image_path, label_path) in zip(image_pathlist, label_pathlist):
        img = cv2.imread(image_path)
        label = read_label_from_txt(label_path)
        if label is None:
            continue
        img, im_scale = preprocess_imgs(img)
        dataset_img_list.append(img)
        g_bboxes.append(label)

    print time.time() - start
    return np.array(dataset_img_list), np.array(g_bboxes)
