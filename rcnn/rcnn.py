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
# from utility.image.data_augmentation.flip import Flip
sys.path.append("/Users/tsujiyuuki/env_python/code/my_code/Data_Augmentation")

"""
・collect dataset of cars
・Preprocessing BBOX and Label for training
・try roi_pooling layer
・Extract ROI using mitmul tools
・NMS
"""

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

def create_optimizer(all_loss, lr=0.001, var_list=None):
    opt = tf.train.AdamOptimizer(lr)
    if var_list is None:
        return opt.minimize(all_loss)
    optimizer = opt.minimize(all_loss, var_list=var_list)
    return optimizer

class RPN_ExtendedLayer(object):
    def __init__(self):
        pass

    def build_model(self, input_layer, use_batchnorm=False, is_training=True, activation=tf.nn.relu, anchors=1):
        self.rpn_conv = convBNLayer(input_layer, use_batchnorm, is_training, 512, 512, 3, 1, name="conv_rpn", activation=activation)
        # shape is [Batch, 2(bg/fg) * 9(anchors=3scale*3aspect ratio)]
        self.rpn_cls = convBNLayer(self.rpn_conv, use_batchnorm, is_training, 512, anchors*2, 1, 1, name="rpn_cls", activation=activation)
        rpn_shape = self.rpn_cls.get_shape().as_list()
        rpn_shape = tf.shape(self.rpn_cls)
        self.rpn_cls = tf.reshape(self.rpn_cls, [rpn_shape[0], rpn_shape[1], rpn_shape[2], anchors, 2])
        self.rpn_cls = tf.nn.softmax(self.rpn_cls, dim=-1)[:, :, :, :, 0]
        self.rpn_cls = tf.reshape(self.rpn_cls, [rpn_shape[0], rpn_shape[1]*rpn_shape[2]*anchors])
        # shape is [Batch, 4(x, y, w, h) * 9(anchors=3scale*3aspect ratio)]
        self.rpn_bbox = convBNLayer(self.rpn_conv, use_batchnorm, is_training, 512, anchors*4, 1, 1, name="rpn_bbox", activation=activation)
        self.rpn_bbox = tf.reshape(self.rpn_bbox, [rpn_shape[0], rpn_shape[1]*rpn_shape[2]*anchors, 4])

class VGG(object):
    def __init__(self):
        pass

    def build_model(self, input_layer, activation=tf.nn.relu, anchors=1):
        self.conv1_1 = convLayer(images, 3, 64, 3, 1, activation=activation, name="conv1_1")
        self.conv1_2 = convLayer(self.conv1_1, 64, 64, 3, 1, activation=activation, name="conv1_2")
        self.pool1 = maxpool2d(self.conv1_2, kernel=2, stride=2, name="pool1")

        self.conv2_1 = convLayer(self.pool1, 64, 128, 3, 1, activation=activation, name="conv2_1")
        self.conv2_2 = convLayer(self.conv2_1, 128, 128, 3, 1, activation=activation, name="conv2_2")
        self.pool2 = maxpool2d(self.conv2_2, kernel=2, stride=2, name="pool2")

        self.conv3_1 = convLayer(self.pool2, 128, 256, 3, 1, activation=activation, name="conv3_1")
        self.conv3_2 = convLayer(self.conv3_1, 256, 256, 3, 1, activation=activation, name="conv3_2")
        self.conv3_3 = convLayer(self.conv3_2, 256, 256, 3, 1, activation=activation, name="conv3_3")
        self.pool3 = maxpool2d(self.conv3_3, kernel=2, stride=2, name="pool3")

        self.conv4_1 = convLayer(self.pool2, 256, 512, 3, 1, activation=activation, name="conv4_1")
        self.conv4_2 = convLayer(self.conv4_1, 512, 512, 3, 1, activation=activation, name="conv4_2")
        self.conv4_3 = convLayer(self.conv4_2, 512, 512, 3, 1, activation=activation, name="conv4_3")
        self.pool4 = maxpool2d(self.conv4_3, kernel=2, stride=2, name="pool4")

        self.conv5_1 = convLayer(self.pool2, 512, 512, 3, 1, activation=activation, name="conv5_1")
        self.conv5_2 = convLayer(self.conv5_1, 512, 512, 3, 1, activation=activation, name="conv5_2")
        self.conv5_3 = convLayer(self.conv5_2, 512, 512, 3, 1, activation=activation, name="conv5_3")

def proposal_layer(feature_map, rpn_model, num_of_rois=num_of_rois):
    return rois, gt_cls, gt_boxes

class FAST_RCNN(object):
    def __init__(self, roi_size):
        self.roi_size = roi_size

    def build_model(self, feature_map, rois, rpn_model, activation=tf.nn.relu):
        """
        **rpn_modelから、実際の大きさまでスケールさせる**
        1. 小さなbounding boxを排除(feature_stride * roi size?)
        2. scoreから6000個を抽出
        3. NMSをかけて、300個以下まで候補を絞る
        ここまでが物体候補領域の抽出
        ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
        4. gt_boxesと候補領域でoverlapsを計算する
        overlapsが0.5以上ならGroundTruth, [0.1, 0.5)ならFalseであるとする　　* ここまでBatchでよい
        ここの計算でReshapeされたROI, 正解Class Label, 正解Regression Label, そのindex番号の計算が行われる
        5. rpn_modelをclass label[?]とregression label[?, 4]にReshapeし、indexで値を取ってくる

        input
        1. Pred class Label
        2. Pred regression Label
        3. GroundTruth class Label
        4. GroundTruth regression Label

        output
        1. 候補領域の計算されたROI（batch number, x, y, w, h), 数は？
        2. 候補領域の正解Class Label(batch number, 2) car or not
        3. 候補領域の正解Regression Label(batch number, 4) x, y, w, h
        　　これも事前に正規化しておく必要があります
        4. Reshapeされたpred class label [?]
        5. Reshapeされたpred regression label [?]

        ここではBack Propは計算されない
        indexのみ計算される  indexのOutputのShapeは、[?]
        ROIs[index]で、これが次の層に伝搬される
        """
# with tf.Session() as sess:
#     a = np.arange(10).reshape(5, 2)
#     b = tf.constant(a)
#     d = tf.nn.top_k(b[:, 0], k=2).indices
#     e = tf.nn.top_k(b[:, 0], k=2).values
#     f, g = sess.run([d, e])
#     print f
#     print g
        # input_layer shape is [Batch, K, A, ]
        self.roi_layer = roi_pooling(feature_map, rois, self.roi_size[0], self.roi_size[1])
        # input_shape [num_of_rois, channel, roi size, roi size]
        # tf.gather(roi_layer)
        self.pool_5 = tf.reshape(roi_layer, [-1, self.roi_size[0]*self.roi_size[1]*512])
        self.fc6 = vgg_fully(self.pool_5, [self.roi_size[0]*self.roi_size[1]*512, 4096], name="fc6", is_training=is_training)
        self.fc7 = vgg_fully(self.fc6, [4096, 4096], name="fc7")
        self.fc8 = vgg_fully(self.fc7, [4096, 6], name="fc8")
        # output shape [num_of_rois, 2]
        self.obj_class = tf.nn.softmax(self.fc8[:, :2], dim=-1)
        # output shape [num_of_rois, 8]
        self.bbox_regression = self.fc8[:, 2:]

def rpn(sess, vggpath=None, image_shape=(300, 300), \
              is_training=None, use_batchnorm=False, activation=tf.nn.relu, anchors=9):
    images = tf.placeholder(tf.float32, [None, None, None, 3])
    phase_train = tf.placeholder(tf.bool, name="phase_traing") if is_training else None

    vgg = VGG()
    vgg.build_model(images)
    with tf.variable_scope("rpn_model"):
        rpn = RPN_ExtendedLayer()
        rpn.build_model(vgg.conv5_3, use_batchnorm=use_batchnorm, is_training=is_training, activation=activation, anchors=anchors)

    if is_training:
        rcnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_model")
        sess.run(tf.variables_initializer(rcnn_vars))
    return vgg.conv5_3, rpn, images, phase_train

def fast_rcnn(sess, feature_map, rpn_model, roi_size=(7, 7), \
              is_training=None, use_batchnorm=False, activation=tf.nn.relu, num_of_rois=128):
    """Model Definition of Fast RCNN
    In thesis, Roi Size is (7, 7), channel is 512
    """
    with tf.variable_scope("fast_rcnn"):
        # py_func
        rois, gt_cls, gt_boxes = proposal_layer(feature_map, rpn_model, num_of_rois=num_of_rois)
        rcnn = FAST_RCNN(roi_size)
        rcnn.build_model(feature_map, rois)

    if is_training:
        rcnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="fast_rcnn")
        sess.run(tf.variables_initializer(rcnn_vars))

    return rcnn, rcnn_vars

def train_rcnn(batch_size, image_dir, label_dir, epoch=101, lr=0.01, feature_shape=(64, 19), \
                  is_training=True, use_batchnorm=False, activation=tf.nn.relu, \
                  scales=np.array([5, 8, 12, 16, 32]), ratios=[0.3, 0.5, 0.8, 1], feature_stride=16):
    import time
    training_epochs = epoch

    with tf.Session() as sess:
        vgg_featuremap, rpn_model, images, phase_train = rpn(sess, vggpath=vggpath, is_training=False, roi_size=(7, 7), \
                                         use_batchnorm=use_batchnorm, activation=activation, anchors=scales.shape[0]*len(ratios))
        saver = tf.train.Saver()
        new_saver = tf.train.import_meta_graph("../rpn/rpn_model40.ckpt.meta")
        last_model = "../rpn/rpn_model40.ckpt"
        saver.restore(sess, last_model)

        rcnn_model, rcnn_vars = fast_rcnn(sess, vgg_featuremap, rpn_model, roi_size=roi_size, activation=activation)

        total_loss, cls_loss, bbox_loss, true_obj_loss, false_obj_loss, g_bboxes, true_index, false_index = rpn_loss(rcnn_model.rcnn_cls, rcnn_model.rcnn_bbox)
        # Only Training RCNN Layer
        optimizer = create_optimizer(total_loss, lr=lr, var_list=rcnn_vars)

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

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from PIL import Image as im
    sys.path.append('/home/katou01/code/grid/DataAugmentation')
    # from resize import resize

    image_dir = "/home/katou01/download/training/image_2/*.png"
    label_dir = "/home/katou01/download/training/label_2/*.txt"
    get_Image_Roi_All(image_dir, label_dir, 80)
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
