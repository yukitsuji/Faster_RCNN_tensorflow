cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=4] pred_boxes,
        np.ndarray[DTYPE_t, ndim=3] gt_boxes):
    """
    Parameters
    ----------
    pred_boxes: (Batch_Size, K, A, 4) ndarray of float
    gt_boxes: (Batch, G, 4) ndarray of float
    Returns
    -------
    overlaps: (Batch_Size, K, A, G) ndarray of overlap between pred_boxes and gt_boxes
    """
    cdef unsigned int Batch_Size = pred_boxes[0]
    cdef unsigned int K = pred_boxes.shape[1]
    cdef unsigned int A = pred_boxes.shape[2]
    cdef unsigned int G = gt_boxes.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=4] overlaps = np.zeros((Batch_Size, K, A, G), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] gt_argmax = np.zeros((Batch_Size, G, 2), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area, max_overlap, max_h, max_w
    cdef DTYPE_t ua
    cdef unsigned int k, a, b, g

    for g in range(G):
        for b in range(Batch_Size):
            box_area = (
                (gt_boxes[b, g, 2] - gt_boxes[b, g, 0] + 1) *
                (gt_boxes[b, g, 3] - gt_boxes[b, g, 1] + 1)
            )
            max_overlap = 0
            max_h = 0
            max_w = 0
            for k in range(K):
                  for a in range(A):
                      if pred_boxes[b, k, a, 2] > 0:
                          iw = (
                              min(pred_boxes[b, k, a, 2], gt_boxes[b, g, 2]) -
                              max(pred_boxes[b, k, a, 0], gt_boxes[b, g, 0]) + 1
                          )
                          if iw > 0:
                              ih = (
                                  min(pred_boxes[b, k, a, 3], gt_boxes[b, g, 3]) -
                                  max(pred_boxes[b, k, a, 1], gt_boxes[b, g, 1]) + 1
                              )
                              if ih > 0:
                                  ua = float(
                                      (pred_boxes[b, k, a, 2] - pred_boxes[b, k, a, 0] + 1) *
                                      (pred_boxes[b, k, a, 3] - pred_boxes[b, k, a, 1] + 1) +
                                      box_area - iw * ih
                                  )
                                  overlaps[b, k, a, k] = iw * ih / ua
                                  if max_overlap < (iw * ih / ua):
                                      max_overlap = iw * ih / ua
                                      max_h = h
                                      max_w = w
            gt_argmax[b, g, 0] = max_h
            gt_argmax[b, g, 1] = max_w
    return overlaps, gt_argmax
