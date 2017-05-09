# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
sys.path.append("../util")
sys.path.append("../cython_util")
sys.path.append("../pretrain")
import glob
import cv2
import numpy as np
# from vgg16 import vgg16
from input_kitti import *
from data_util import *
from parse_xml import parseXML
from vgg16_vehicle import Vgg16 as Vgg
import tensorflow as tf
from network_util import *
from bbox_overlap import bbox_overlaps
from remove_extraboxes import remove_extraboxes
from bool_anchors_inside_image import batch_inside_image
from generate_anchors import generate_anchors
# from utility.image.data_augmentation.flip import Flip
# sys.path.append("/Users/tsujiyuuki/env_python/code/my_code/Data_Augmentation")


"""Flow of Fast RCNN
###############################################################################
In this state, Create Input Images and ROI Labels

1. input batch images and GroundTruth BBox from datasets *folder name, batch size
   Image shape is [batch size, width, height, channel], tf.float32, vgg normalized, bgr
   Bounding Box shape is [batch size, center_x, center_y, width, height]

2. get candicate bounding box from images.

   # Implemented
3. resize input images to input size   *size of resize    if needed.
   if this operation was done, you should adjust bounding box according to it.
   Both of Candicate and GroundTruth Bounding Boxes.
   In thesis, Image size is in [600, 1000]
   In this Implemention, input image has dynamic shape between [600, 1000]

4. convert candicate bounding box to ROI label.

5. calculate IOU between ROI label and GroundTruth label.
   IOU is Intersection Over Union.

6. Select Bounding Box from IOU.
   IOU > 0.5 is correct label, IOU = [0.1 0.5) is a false label(background).
   Correct Label is 25%, BackGround Label is 75%.
   Number of Label is 128, Batch Size is 2, so each image has 64 ROIs

###############################################################################
In this stage, Calculate Loss

7. Input data to ROI Pooling Layer is Conv5_3 Feature Map and ROIs
   Input shape is Feature map (batch, width, height, 512), ROIs (Num of ROIs, 5)
   ROIs, ex:) [0, left, height, right, bottom]. First Element is the index of batch

8. Through ROI Pooling Layer, Output Shape is [Num of ROIs, 7, 7, 512]

9. Reshape it to [Num of ROIs, -1], and then connect to Fully Connected Layer.

10.Output Layer has two section, one is class prediction, the other is its bounding box prediction.
   class prediction shape is [Num of ROIs, Num of Class + 1]
   bounding box prediction shape is [Num of ROIs, 4 * (Num of Class + 1)]

11.Loss Function
   Regularize bounding box value [center_x, center_y, w, h] into
   [(GroundTruth x - pred_x) / pred_w, (GroundTruth y - pred_y) / pred_h, log(GroundTruth w / pred_w), log(GroundTruth h / pred_h)]
   Class prediction is by softmax with loss.
   Bounding Box prediction is by smooth_L1 loss
###############################################################################
In this stage, Describe Datasets.
1. PASCAL VOC2007
2. KITTI Datasets
3. Udacity Datasets
"""

# TODO: datasetsを丸ごとメモリに展開できるか。Generatorを用いるか。


def create_optimizer(all_loss, lr=0.001):
    opt = tf.train.AdamOptimizer(lr)
    optimizer = opt.minimize(all_loss)
    return optimizer

class RPN_ExtendedLayer(object):
    def __init__(self):
        pass

    def build_model(self, input_layer, use_batchnorm=False, is_training=True, atrous=False, \
                    rate=1, activation=tf.nn.relu, implement_atrous=False, lr_mult=1, anchors=1):
        self.rpn_conv = convBNLayer(input_layer, use_batchnorm, is_training, 512, 512, 3, 1, name="conv_rpn", activation=activation)
        # shape is [Batch, 2(bg/fg) * 9(anchors=3scale*3aspect ratio)]
        self.rpn_cls = convBNLayer(self.rpn_conv, use_batchnorm, is_training, 512, anchors*2, 1, 1, name="rpn_cls", activation=activation)
        rpn_shape = self.rpn_cls.get_shape().as_list()
        rpn_shape = tf.shape(self.rpn_cls)
        self.rpn_cls = tf.reshape(self.rpn_cls, [rpn_shape[0], rpn_shape[1], rpn_shape[2], anchors, 2])
        self.rpn_cls = tf.nn.softmax(self.rpn_cls, dim=-1)
        self.rpn_cls = tf.reshape(self.rpn_cls, [rpn_shape[0], rpn_shape[1]*rpn_shape[2], anchors, 2])
        # shape is [Batch, 4(x, y, w, h) * 9(anchors=3scale*3aspect ratio)]
        self.rpn_bbox = convBNLayer(self.rpn_conv, use_batchnorm, is_training, 512, anchors*4, 1, 1, name="rpn_bbox", activation=activation)
        self.rpn_bbox = tf.reshape(self.rpn_bbox, [rpn_shape[0], rpn_shape[1]*rpn_shape[2], anchors, 4])

def rpn(sess, vggpath=None, image_shape=(300, 300), \
              is_training=None, use_batchnorm=False, activation=tf.nn.relu, anchors=9):
    images = tf.placeholder(tf.float32, [None, None, None, 3])
    phase_train = tf.placeholder(tf.bool, name="phase_traing") if is_training else None

    vgg = Vgg(vgg16_npy_path=vggpath)
    vgg.build_model(images)

    with tf.variable_scope("rpn_model") as scope:
        rpn_model = RPN_ExtendedLayer()
        rpn_model.build_model(vgg.conv4_3, use_batchnorm=use_batchnorm, \
                                   is_training=phase_train, activation=activation, anchors=anchors)

    if is_training:
        initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_model")
        sess.run(tf.variables_initializer(initialized_var))
    return rpn_model, images, phase_train

def smooth_L1(x):
    l2 = 0.5 * (x**2.0)
    l1 = tf.abs(x) - 0.5

    condition = tf.less(tf.abs(x), 1.0)
    loss = tf.where(condition, l2, l1)
    return loss

def rpn_loss(rpn_cls, rpn_bbox):
    """Calculate Class Loss and Bounding Regression Loss.

    # Args:
        obj_class: Prediction of object class. Shape is [ROIs*Batch_Size, 2]
        bbox_regression: Prediction of bounding box. Shape is [ROIs*Batch_Size, 4]
    """
    rpn_shape = rpn_cls.get_shape().as_list()
    g_bbox = tf.placeholder(tf.float32, [rpn_shape[0], rpn_shape[1], rpn_shape[2], 4])
    true_index = tf.placeholder(tf.float32, [rpn_shape[0], rpn_shape[1], rpn_shape[2]])
    false_index = tf.placeholder(tf.float32, [rpn_shape[0], rpn_shape[1], rpn_shape[2]])
    elosion = 0.00001
    true_obj_loss = -tf.reduce_sum(tf.multiply(tf.log(rpn_cls[:, :, :, 0]+elosion), true_index))
    false_obj_loss = -tf.reduce_sum(tf.multiply(tf.log(rpn_cls[:, :, :, 1]+elosion), false_index))
    obj_loss = tf.add(true_obj_loss, false_obj_loss)
    cls_loss = tf.div(obj_loss, 16) # L(cls) / N(cls) N=batch size

    bbox_loss = smooth_L1(tf.subtract(rpn_bbox, g_bbox))
    bbox_loss = tf.reduce_sum(tf.multiply(tf.reduce_sum(bbox_loss, 3), true_index))
    bbox_loss = tf.multiply(tf.div(bbox_loss, 1197), 100) # rpn_shape[1]*rpn_shape[2]
    # bbox_loss = bbox_loss / rpn_shape[1]

    total_loss = tf.add(cls_loss, bbox_loss)
    return total_loss, cls_loss, bbox_loss, true_obj_loss, false_obj_loss, g_bbox, true_index, false_index


def create_Labels_For_Loss(gt_boxes, feat_stride=16, feature_shape=(64, 19), \
                           scales=np.array([8, 16, 32]), ratios=[0.5, 0.8, 1], \
                           image_size=(300, 1000)):
    """This Function is processed before network input
    Number of Candicate Anchors is Feature Map width * heights
    Number of Predicted Anchors is Batch Num * Feature Map Width * Heights * 9
    """
    width = feature_shape[0]
    height = feature_shape[1]
    batch_size = gt_boxes.shape[0]
    # shifts is the all candicate anchors(prediction of bounding boxes)
    center_x = np.arange(0, height) * feat_stride
    center_y = np.arange(0, width) * feat_stride
    center_x, center_y = np.meshgrid(center_x, center_y)
    # Shape is [Batch, Width*Height, 4]
    centers = np.zeros((batch_size, width*height, 4))
    centers[:] = np.vstack((center_x.ravel(), center_y.ravel(),
                        center_x.ravel(), center_y.ravel())).transpose()
    A = scales.shape[0] * len(ratios)
    K = width * height # width * height
    anchors = np.zeros((batch_size, A, 4))
    anchors = generate_anchors(scales=scales, ratios=ratios) # Shape is [A, 4]

    candicate_anchors = centers.reshape(batch_size, K, 1, 4) + anchors # [Batch, K, A, 4]

    # shape is [B, K, A]
    is_inside = batch_inside_image(candicate_anchors, image_size[1], image_size[0])

    # candicate_anchors: Shape is [Batch, K, A, 4]
    # gt_boxes: Shape is [Batch, G, 4]
    # true_index: Shape is [Batch, K, A]
    # false_index: Shape is [Batch, K, A]
    candicate_anchors, true_index, false_index = bbox_overlaps(
        np.ascontiguousarray(candicate_anchors, dtype=np.float),
        is_inside,
        gt_boxes)

    for i in range(batch_size):
        true_where = np.where(true_index[i] == 1)
        num_true = len(true_where[0])

        if num_true > 64:
            select = np.random.choice(num_true, num_true - 64, replace=False)
            num_true = 64
            batch = np.ones((select.shape[0]), dtype=np.int) * i
            true_where = remove_extraboxes(true_where[0], true_where[1], select, batch)
            true_index[true_where] = 0

        false_where = np.where(false_index[i] == 1)
        num_false = len(false_where[0])
        select = np.random.choice(num_false, num_false - (128-num_true), replace=False)
        batch = np.ones((select.shape[0]), dtype=np.int) * i
        false_where = remove_extraboxes(false_where[0], false_where[1], select, batch)
        false_index[false_where] = 0

    return candicate_anchors, true_index, false_index

def train_rpn(batch_size, image_dir, label_dir, epoch=101, lr=0.01, feature_shape=(64, 19), \
                  vggpath="../pretrain/vgg16.npy", use_batchnorm=False, activation=tf.nn.relu, \
                  scales=np.array([5, 8, 12, 16, 32]), ratios=[0.3, 0.5, 0.8, 1], feature_stride=16):
    import time
    training_epochs = epoch

    with tf.Session() as sess:
        model, images, phase_train = rpn(sess, vggpath=vggpath, is_training=True, \
                                         use_batchnorm=use_batchnorm, activation=activation, anchors=scales.shape[0]*len(ratios))
        total_loss, cls_loss, bbox_loss, true_obj_loss, false_obj_loss, g_bboxes, true_index, false_index = rpn_loss(model.rpn_cls, model.rpn_bbox)
        optimizer = create_optimizer(total_loss, lr=lr)
        init = tf.global_variables_initializer()
        sess.run(init)

        image_pathlist, label_pathlist = get_pathlist(image_dir, label_dir)
        for epoch in range(training_epochs):
            for batch_images, batch_labels in generator__Image_and_label(image_pathlist, label_pathlist, batch_size=batch_size):
                start = time.time()
                candicate_anchors, batch_true_index, batch_false_index = create_Labels_For_Loss(batch_labels, feat_stride=feature_stride, \
                    feature_shape=(batch_images.shape[1]//feature_stride +1, batch_images.shape[2]//feature_stride), \
                    scales=scales, ratios=ratios, image_size=batch_images.shape[1:3])
                print "batch time", time.time() - start
                print batch_true_index[batch_true_index==1].shape
                print batch_false_index[batch_false_index==1].shape

                sess.run(optimizer, feed_dict={images:batch_images, g_bboxes: candicate_anchors, true_index:batch_true_index, false_index:batch_false_index})
                tl, cl, bl, tol, fol = sess.run([total_loss, cls_loss, bbox_loss, true_obj_loss, false_obj_loss], feed_dict={images:batch_images, g_bboxes: candicate_anchors, true_index:batch_true_index, false_index:batch_false_index})
                print("Epoch:", '%04d' % (epoch+1), "total loss=", "{:.9f}".format(tl))
                print("Epoch:", '%04d' % (epoch+1), "closs loss=", "{:.9f}".format(cl))
                print("Epoch:", '%04d' % (epoch+1), "bbox loss=", "{:.9f}".format(bl))
                print("Epoch:", '%04d' % (epoch+1), "true loss=", "{:.9f}".format(tol))
                print("Epoch:", '%04d' % (epoch+1), "false loss=", "{:.9f}".format(fol))
    print("Optimization Finished")

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from PIL import Image as im
    sys.path.append('/home/katou01/code/grid/DataAugmentation')
    # from resize import resize

    image_dir = "/home/katou01/download/training/image_2/*.png"
    label_dir = "/home/katou01/download/training/label_2/*.txt"
    # import time
    train_rpn(4, image_dir, label_dir, epoch=20, lr=0.001, \
               scales=np.array([6, 8, 10, 12, 14, 16, 20, 32]), ratios=[0.4,  0.6, 0.8, 1.0], feature_stride=8)
    # image_pathlist, label_pathlist = get_pathlist(image_dir, label_dir)
    # for images, labels in generator__Image_and_label(image_pathlist, label_pathlist, batch_size=32):
    #     start = time.time()
    #     # images, labels = get_ALL_Image(image_pathlist, label_pathlist)
    #     candicate_anchors, true_index, false_index = create_Labels_For_Loss(labels, feat_stride=16, feature_shape=(64, 19), \
    #                                scales=np.array([5,  8, 12, 16, 32]), ratios=[0.3, 0.5, 0.8, 1], \
    #                                image_size=(302, 1000))
    #     print "batch time", time.time() - start
    #     print candicate_anchors.shape, true_index.shape, false_index.shape
    # # images, labels = get_ALL_Image(image_pathlist, label_pathlist)
    # candicate_anchors, true_index, false_index = create_Labels_For_Loss(labels, feat_stride=16, feature_shape=(64, 19), \
    #                            scales=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32]), ratios=[0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.2], \
    #                            image_size=(302, 1000))
