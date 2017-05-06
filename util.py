# -*- coding: utf-8 -*-

import sys
import glob
import cv2
import numpy as np
# from vgg16 import vgg16
from input_kitti import *
from parse_xml import parseXML
from bbox_transform import *
from base_vgg16 import Vgg16
import tensorflow as tf
from bbox_overlap import bbox_overlaps

def create_labels(resized_images, resize_scales, feature_scale=1./16):
    """create labels for classification and regression
    1. get bbox from resized images
    2. from bbox, create input labels for regression
    3. get GroundTruth Bounding Boxes
    4. calculate IOU for training
    5. divide labels into training sets and trush
    6.
    """
    return labels

def create_rois(labels, feature_scale=1./16):
    """create rois from labels"""
    return rois

def nms():
    return bboxes

def process(image_dir, label_dir, num_of_rois, batch_size, min_size):
    # model Definition
    # loss function
    dataset_img_list, dataset_pred_bbox_list, g_bboxes, get_Image_Roi_All(image_dir, label_dir, min_size)
    # batch_imgs, batch_rois, batch_g_bboxes = select_inputs_from_datasets(dataset_img_list, dataset_pred_bbox_list, g_bboxes, batch_size)
    for batch_imgs, batch_rois, batch_g_bboxes in select_inputs_from_datasets(dataset_img_list, dataset_pred_bbox_list, g_bboxes, batch_size):
        pass
        # training
        # test
        # validation

def get_Image_Roi_All(image_dir, label_dir, min_size):
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
    g_bboxes = 0 #load_for_detection(label_dir) #TODO: [Datasets, x, y, w, h]
    dataset_img_list = [] # len(dataset_img_list) == Number of Datasets Images
    dataset_pred_bbox_list = [] # len(dataset_pred_bbox_list) == Number of (num_of_rois * num of images)
    # Preprocess Ground Truth ROIs. shape is [Num of ROIs * batch_size, x, y, w, h, 0, 1]
    g_bboxes = []
    # shape is [batch_channel, x, y, w, h]
    image_pathlist = glob.glob(image_dir)
    label_pathlist = glob.glob(label_dir)
    image_pathlist.sort()
    label_pathlist.sort()

    for index, (image_path, label_path) in enumerate(zip(image_pathlist, label_pathlist)):
        if index == 10:
            break
        img = cv2.imread(image_path)
        label = read_label_from_txt(label_path)
        if label is None:
            continue
        # ここでは、IOUを計算していないので、予測のbounding boxは絞らない
        # なので、数多くのbounding boxが存在していることになるが、メモリが許す限り確保する
        p_bbox_candicate = pred_bboxes(img, min_size, index)
        img, im_scale = preprocess_imgs(img)
        p_bbox_candicate = unique_bboxes(p_bbox_candicate, im_scale, feature_scale=1./16)
        overlaps = bbox_overlaps(p_bbox_candicate[:, 1:], label)
        print label
        print p_bbox_candicate[0]
        print overlaps[overlaps > 0.5]
        print overlaps.shape
        print
        dataset_img_list.append(img)
        dataset_pred_bbox_list.append(p_bbox_candicate)
        g_bboxes.append(label)

    dataset_pred_bbox_list = np.array(dataset_pred_bbox_list)
    g_bboxes = np.array(g_bboxes)
    print dataset_img_list[1].shape, dataset_pred_bbox_list[0].shape, g_bboxes[0].shape
    print dataset_pred_bbox_list[1].shape
    print dataset_pred_bbox_list[2].shape
    g_bboxes = create_bbox_regression_label(dataset_pred_bbox_list, g_bboxes)
    return np.array(dataset_img_list), np.array(dataset_pred_bbox_list), g_bboxes

def select_inputs_from_datasets(dataset_img_list, dataset_pred_bbox_list, g_bboxes, batch_size):
    """
    # Args:
        dataset_img_list      (ndarray): ndarray Images in datasets.
        dataset_pred_bbox_list(ndarray): rescaled bbox Label [0, x, y, w, h]
                                         shape is [batch, num_of_rois, 5]
        g_bboxes              (ndarray): GroundTruth Bounding Box with Class Label
                                         shape is [batch, 6*max_label_num]
                                         label is [x, y, w, h, car, background]
        batch_size                (int): batch size for training
    # Returns:
        batch_imgs    (ndarray): input batch images for Network. Shape is [Batch Size, shape]
        batch_p_bboxes(ndarray): input ROIs for Network. Shape is [Num of ROIs*Batch size]
        batch_g_bboxes(ndarray): input GroundTruth Bounding Box for Network.
                                 Shape is [Num of ROIs*Batch Size]
    """
    perm = np.random.permutation(len(dataset_img_list))
    batches = [perm[i * batch_size:(i + 1) * batch_size] \
                   for i in range(len(dataset_img_list) // batch_size)]
    for batch in batches:
        batch_imgs = dataset_img_list[batch]
        batch_p_bboxes = dataset_pred_bbox_list[batch]
        batch_g_bboxes = g_bboxes[batch]
        # この時点でbatch_p_bboxes, g_bboxesは、batch毎にListでまとめられていそう？　#TODO
        # TODO: Batch毎にLabelの形にする。それをcalculate IOUに入れて、最終的な形をvstackすれば全体のLabelが得られる

        # Flip Conversion
        # batch_imgs, batch_p_bboxes, batch_g_bboxes = flip_conversion(batch_imgs, batch_p_bboxes, batch_g_bboxes)
        batch_imgs = convert_imgslist_to_ndarray(batch_imgs)
        # calculate IOU between pred_roi_candicate, ground truth bounding box
        # この時点でbatch_g_bboxesはLabelの形になっていると想定
        batch_p_bboxes, batch_g_bboxes = calculate_IOU(batch_p_bboxes, batch_g_bboxes)
        yield batch_imgs, batch_rois, batch_g_bboxes

def convert_pred_bbox_to_roi(batch_bbox, feature_scale=1./16):
    pass

def calculate_IOU(batch_roi, batch_g_bboxes, fg_thres=0.5, bg_thres_max=0.5, bg_thres_min=0.1):
    """各画像の全ての車のラベルに対して、IOUを計算する
    そのために、batch_roi, batch_g_bboxesをforループで回し、
    """
    area = batch_g_bboxes[:, 3] * batch_g_bboxes[: 4]
    w = np.maximum(batch_roi[:, 0], batch_g_bboxes[:, 0]) - np.minimum(batch_roi[:, 1], batch_g_bboxes[:, 1])
    w_id = np.where(w > 0)[0]
    h = np.minimum(batch_roi[w_id][:, 0], batch_g_bboxes[w_id][:, 0]) - np.minimum(batch_roi[w_id][:, 1], batch_g_bboxes[w_id][:, 1])
    h_id = np.where(h > 0)[0]
    IOU = float(w[w_id][h_id] * h[w_id][h_id]) / area[w_id][h_id]
    fg_rois = np.where(IOU >= fg_thres)[0]
    bg_rois1 = np.where(IOU < bg_thres_max)[0]
    bg_rois2 = np.where(IOU[bg_rois] >= bg_thres_min)[0]
    fg_index = w_id[h_id][fg_rois]
    bg_index = w_id[h_id][bg_rois1][bg_rois2]
    index = np.hstack((fg_index, bg_index))
    return batch_rois[index], batch_g_bboxes[index]

def convert_imgslist_to_ndarray(images):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).

    In this stage, the shape of images are different
    """
    max_shape = np.array([im.shape for im in images]).max(axis=0)
    num_images = len(images)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = images[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    return blob

def flip_conversion(batch_imgs, batch_rois, batch_g_bboxes, batch_size):
    return batch_imgs, batch_rois, batch_g_bboxes

def preprocess_imgs(im, pixel_means=np.array([103.939, 116.779, 123.68]), target_size=600, max_size=1000):
    """Mean subtract and scale an image for use in a blob.
    If you want to Data Augmentation, please edit this function
    """
    im = im.astype(np.float32, copy=False)
    # if np.random.randint(2):
    #     im = im[:, ::-1]
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def data_generator(imgs, rois, labels):
    """data generator for network inputs"""
    yield batch_x, batch_rois, batch_labels

def unique_bboxes(rects, im_scale, feature_scale=1./16):
    """Get Bounding Box from Original Image.

    # Args:
        orig_img   (ndarray): original image. 3 dimensional array.
        min_size     (tuple): minimum size of bounding box.
        feature_scale(float): scale of feature map. 2 ** (num of pooling layer)

    """
    rects *= im_scale
    v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    hashes = np.round(rects * feature_scale).dot(v)
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    rects = rects[index, :]
    return rects

def pred_bboxes(orig_img, min_size, index):
    rects = []
    dlib.find_candidate_object_locations(orig_img, rects, min_size=min_size)
    rects = [[0, d.left(), d.top(), d.right(), d.bottom()] for d in rects]
    rects = np.asarray(rects, dtype=np.float64)
    return rects
