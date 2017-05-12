cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from nms cimport bbox_transform_inv

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_transform_inv_clip(
        np.ndarray[DTYPE_t, ndim=3] anchors,
        np.ndarray[DTYPE_t, ndim=3] rpn_bbox,
        unsigned int image_width,
        unsigned int image_height):
    """
    Parameters
    ----------
    anchors: (Batch_Sizes, K*A, 4) ndarray of float
    rpn_bbox: (Batch_Size, K*A, 4) ndarray of float
    -------
    """
    cdef unsigned int B = anchors.shape[0]
    cdef unsigned int KA = anchors.shape[1]
    cdef DTYPE_t ex_width, ex_height, ex_center_x, ex_center_y, gt_width, gt_height, gt_center_x, gt_center_y
    cdef unsigned int ka, b

    for b in range(Batch_Size):
        for ka in range(KA):
            ex_width = anchors[b, ka, 2] - anchors[b, ka, 0] + 1
            ex_height = anchors[b, ka, 3] - anchors[b, ka, 1] + 1
            ex_center_x = anchors[b, ka, 0] + ex_width / 2.0
            ex_center_y = anchors[b, ka, 1] + ex_height / 2.0

            pred_center_x = rpn_bbox[b, ka, 0] * ex_width + ex_center_x
            pred_center_y = rpn_bbox[b, ka, 1] * ex_height + ex_center_y
            pred_width = exp(rpn_bbox[b, ka, 2]) * ex_width
            pred_height = exp(rpn_bbox[b, ka, 3]) * ex_height

            anchors[b, ka, 0] = max(pred_center_x - pred_width / 2.0, 0)
            anchors[b, ka, 1] = max(pred_center_y - pred_height / 2.0, 0)
            anchors[b, ka, 2] = min(pred_center_x + pred_width / 2.0, image_width-1)
            anchors[b, ka, 3] = min(pred_center_y + pred_height / 2.0, image_height-1)


    return anchors
