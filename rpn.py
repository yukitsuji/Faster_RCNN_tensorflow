# -*- coding: utf-8 -*-

import sys
import glob
import cv2
import dlib
import numpy as np
# from vgg16 import vgg16
from input_kitti import *
from util import *
from parse_xml import parseXML
from base_vgg16 import Vgg16
import tensorflow as tf
from network_util import *
# from utility.image.data_augmentation.flip import Flip
sys.path.append("/Users/tsujiyuuki/env_python/code/my_code/Data_Augmentation")

"""
・collect dataset of cars
・Preprocessing BBOX and Label for training
・try roi_pooling layer
・Extract ROI using mitmul tools
・NMS
"""

"""Flow of Faster RCNN
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
def rpn_loss(obj_class, bbox_regression, g_obj_class, g_bbox_regression):
    """Calculate Class Loss and Bounding Regression Loss.

    # Args:
        obj_class: Prediction of object class. Shape is [ROIs*Batch_Size, 2]
        bbox_regression: Prediction of bounding box. Shape is [ROIs*Batch_Size, 4]
    """
    pass

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
        # shape is [Batch, 4(x, y, w, h) * 9(anchors=3scale*3aspect ratio)]
        self.rpn_bbox = convBNLayer(self.rpn_conv, use_batchnorm, is_training, 512, 36, 1, 1, name="rpn_bbox", activation=activation)

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

def create_gtlabel(g_labels, feat_stride=16):
    """
    Number of Anchors is 128
    """
    # shifts is the all candicate anchors(prediction of bounding boxes)
    # Not ROIs
    shift_x = np.arange(0, width) * self._feat_stride
    shift_y = np.arange(0, height) * self._feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

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
    images, labels = get_ALL_Image(image_dir, label_dir)
    print images.shape, labels.shape
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
