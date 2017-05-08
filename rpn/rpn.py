# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/katou01/code/Faster_RCNN_tensorflow")
sys.path.append("/home/katou01/code/Faster_RCNN_tensorflow/util")
sys.path.append("/home/katou01/code/Faster_RCNN_tensorflow/cython_util")
import glob
import cv2
import numpy as np
# from vgg16 import vgg16
from input_kitti import *
from data_util import *
from parse_xml import parseXML
from base_vgg16 import Vgg16
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
                    rate=1, activation=tf.nn.relu, implement_atrous=False, lr_mult=1):
        self.rpn_conv = convBNLayer(input_layer, use_batchnorm, is_training, 512, 512, 3, 1, name="conv_rpn", activation=activation)
        # shape is [Batch, 2(bg/fg) * 9(anchors=3scale*3aspect ratio)]
        self.rpn_cls = convBNLayer(self.rpn_conv, use_batchnorm, is_training, 512, 18, 1, 1, name="rpn_cls", activation=activation)
        self.rpn_cls = tf.reshape(self.rpn_cls, [rpn_cls_shape[0], rpn_cls_shape[1], rpn_cls_shape[2], 9, 2])
        self.rpn_cls = tf.nn.softmax(self.rpn_cls, dim=-1)
        # shape is [Batch, 4(x, y, w, h) * 9(anchors=3scale*3aspect ratio)]
        self.rpn_bbox = convBNLayer(self.rpn_conv, use_batchnorm, is_training, 512, 36, 1, 1, name="rpn_bbox", activation=activation)
        self.rpn_bbox = tf.reshpae(self.rpn_bbox, [rpn_cls_shape[0], rpn_cls_shape[1], rpn_cls_shape[2], 9, 4])

def rpn(sess, vggpath=None, image_shape=(300, 300), \
              is_training=None, use_batchnorm=False, activation=tf.nn.relu):
    images = tf.placeholder(tf.float32, [None, None, None, 3])
    phase_train = tf.placeholder(tf.bool, name="phase_traing") if is_training else None

    vgg = Vgg(vgg16_npy_path=vggpath)
    vgg.build_model(images)

    with tf.variable_scope("rpn_model") as scope:
        rpn_model = RPN_ExtendedLayer()
        rpn_model.build_model(vgg.conv5_3, use_batchnorm=use_batchnorm, \
                                   is_training=phase_train, activation=activation)

    if is_training:
        initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_model")
        sess.run(tf.variables_initializer(initialized_var))
    return rpn_model, images, phase_train

def create_predlabel(g_labels, image_shape):
    """Create class label and bbox label
    Compared ground truth with prediction by the roi scales?

    # Returns:
        cls_labels (ndarray): shape is [batch, w/16, h/16, 9, 2]
        bbox_labels(ndarray): shape is [batch, w/16, h/16, 9, 4]
    """
    # g_rois =
    candicate_cls = np.zeros((batch_num, width, height, 9, 2))
    pass

def smooth_L1(x):
    l2 = 0.5 * (x**2.0)
    l1 = tf.abs(x) - 0.5

    condition = tf.less(tf.abs(x), 1.0)
    loss = tf.where(condition, l2, l1)
    return loss

def rpn_loss(rpn_cls, rpn_bbox, g_bbox_regression, true_index, false_index):
    """Calculate Class Loss and Bounding Regression Loss.

    # Args:
        obj_class: Prediction of object class. Shape is [ROIs*Batch_Size, 2]
        bbox_regression: Prediction of bounding box. Shape is [ROIs*Batch_Size, 4]
    """
    rpn_cls = tf.reshape(rpn_cls, [-1, 2])
    rpn_bbox = tf.reshape(rpn_bbox, [-1, 4])

    true_obj_loss = -tf.reduce_sum(tf.multiply(rpn_cls[:, :, :, :, 0], true_index))
    false_obj_loss = -tf.reduce_sum(tf.multiply(rpn_cls[:, :, :, :, 1], false_index))
    obj_loss = tf.add(true_obj_loss, false_obj_loss)
    cls_loss = tf.div(smooth_L1(rpn_bbox, g_bbox_regression), true_obj_loss.get_shape().as_list()[0]) # L(cls) / N(cls) N=batch size

    bbox_loss = smooth_L1(tf.subtract(rpn_bbox, g_bbox_regression))
    bbox_loss = tf.reduce_sum(tf.multiply(bbox_loss, true_index))
    bbox_loss = tf.div(bbox_loss, ANCHORS_NUM)# TODO

    total_loss = tf.add(cls_loss, bbox_loss)
    return total_loss, cls_loss, bbox_loss


def create_Labels_For_Loss(gt_boxes, feat_stride=16, feature_shape=(64, 19), \
                           scales=np.array([8, 16, 32]), ratios=[0.5, 0.8, 1], \
                           image_size=(300, 1000)):
    """This Function is processed before network input
    Number of Candicate Anchors is Feature Map width * heights
    Number of Predicted Anchors is Batch Num * Feature Map Width * Heights * 9
    """
    import time
    func_start = time.time()
    width = feature_shape[0]
    height = feature_shape[1]
    batch_size = gt_boxes.shape[0]
    # shifts is the all candicate anchors(prediction of bounding boxes)
    center_x = np.arange(0, width) * feat_stride
    center_y = np.arange(0, height) * feat_stride
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

    print "time", time.time() - func_start
    return candicate_anchors, true_index, false_index

def create_ROIs_For_RCNN(model, g_labels, feat_stride=16):
    """This Function is processed before network input
    Number of Candicate Anchors is Feature Map width * heights
    Number of Predicted Anchors is Batch Num * Feature Map Width * Heights * 9
    """
    width = network_width
    height = network_height
    # shifts is the all candicate anchors(prediction of bounding boxes)
    # Not ROIs
    shift_x = np.arange(0, width) * self._feat_stride
    shift_y = np.arange(0, height) * self._feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # Shape is [Width*Height, 4]
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0] # width * height

    anchors = generate_anchors(scale=[8, 16, 32], ratio=[0.5, 1., 2.]) # Shape is [1, A, 4]
    A = 9 # [9, 4]
    solid_anchors = shifts.reshape(-1, 1, 4) + anchors.reshape(1, -1, 4)
    solid_anchors = solid_anchors.reshape(-1, 4) # (K * A, 4)
    # diff_bbox is the difference of bbox Regression
    # Shape is [Batch, w/16, h/16, 9, 4]
    diff_bbox = diff_bbox.reshape(-1, 4)
    cls_map = cls_map.reshape(-1, 2)

    # solid_anchors, diff_bbox Dtype = np.float64
    proposal = create_proposal_from_pred(solid_anchors, diff_bbox)

def train_rpn(batch_size, image_dir, label_dir, epoch=101, label_type="txt", lr=0.01, \
                  vggpath="./vgg16.npy", use_batchnorm=False, activation=tf.nn.relu, \
                  min_size=1000):
    training_epochs = epoch

    with tf.Session() as sess:
        model, images, phase_train = rpn(sess, vggpath=vggpath, is_training=True, \
                                         use_batchnorm=use_batchnorm, activation=activation)
        total_loss, cls_loss, bbox_loss = rpn_loss(model)
        optimizer = create_optimizer(total_loss, lr=lr)
        init = tf.global_variables_initializer()
        sess.run(init)

        input_images, cls_labels, bbox_labels = func(image_dir, label_dir, min_size=min_size)

        for epoch in range(training_epochs):
            for (batch_x, batch_cls, batch_bbox) in zip(input_images, cls_labels, bbox_labels):
                sess.run(optimizer, feed_dict={images:batch_x, g_cls:batch_cls, g_bbox:batch_bbox})
                tl = sess.run(total_loss, feed_dict={images:batch_x, g_cls:batch_cls, g_bbox:batch_bbox})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(tl))
    print("Optimization Finished")

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from PIL import Image as im
    sys.path.append('/home/katou01/code/grid/DataAugmentation')
    # from resize import resize

    image_dir = "/home/katou01/download/training/image_2/*.png"
    label_dir = "/home/katou01/download/training/label_2/*.txt"
    import time
    start = time.time()
    images, labels = get_ALL_Image(image_dir, label_dir)
    print "labels"
    print labels.shape
    print "images"
    print images.shape
    candicate_anchors, true_index, false_index = create_Labels_For_Loss(labels, feat_stride=16, feature_shape=(64, 19), \
                               scales=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32]), ratios=[0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.2], \
                               image_size=(302, 1000))
    print time.time() - start
    print images[0].shape, labels.shape

    print true_index[true_index==1].shape
    print false_index[false_index==1].shape

    print true_index[0, true_index[0]==1].shape
    print false_index[0, false_index[0]==1].shape
    #
    # image = im.open("./test_images/test1.jpg")
    # image = np.array(image, dtype=np.float32)
    # new_image = image[np.newaxis, :]
    # batch_image = np.vstack((new_image, new_image))
    # batch_image = resize(batch_image, size=(300, 300))
    #
    # with tf.Session() as sess:
    #     model = ssd_model(sess, batch_image, activation=None, atrous=False, rate=1, implement_atrous=False)
    #     print(vars(model))
    #     # tf.summary.scalar('model', model)
