cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport log

DTYPE = np.float
ctypedef np.float_t DTYPE_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=4] anchors,
        np.ndarray[DTYPE_int_t, ndim=3] is_inside,
        object gt_boxes):
    """
    Parameters
    ----------
    anchors: (Batch_Size, K, A, 4) ndarray of float
    is_inside: (Batch_Size, K, A) ndarray of int
    gt_boxes: (Batch, G, 4) ndarray of float
    Returns
    -------
    """
    cdef unsigned int Batch_Size = anchors.shape[0]
    cdef unsigned int K = anchors.shape[1]
    cdef unsigned int A = anchors.shape[2]
    cdef unsigned int G
    cdef np.ndarray[DTYPE_t, ndim=4] overlaps
    cdef np.ndarray[DTYPE_int_t, ndim=3] true_index = np.zeros((Batch_Size, K, A), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=3] false_index = np.zeros((Batch_Size, K, A), dtype=DTYPE_int)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef DTYPE_t max_overlap
    cdef DTYPE_t ex_width, ex_height, ex_center_x, ex_center_y, gt_width, gt_height, gt_center_x, gt_center_y
    cdef unsigned int k, a, b, g, max_k, max_a, max_g

    max_g = 0
    for b in range(Batch_Size):
        if max_g < gt_boxes[b].shape[0]:
            max_g = gt_boxes[b].shape[0]

    overlaps = np.zeros((Batch_Size, K, A, max_g))

    for b in range(Batch_Size):
        G = gt_boxes[b].shape[0]
        for g in range(G):
            box_area = (
                (gt_boxes[b][g, 2] - gt_boxes[b][g, 0] + 1) *
                (gt_boxes[b][g, 3] - gt_boxes[b][g, 1] + 1)
            )
            max_overlap = 0
            max_k = 0
            max_a = 0
            for k in range(K):
                  for a in range(A):
                      if is_inside[b, k, a] == 1:
                          iw = (
                              min(anchors[b, k, a, 2], gt_boxes[b][g, 2]) -
                              max(anchors[b, k, a, 0], gt_boxes[b][g, 0]) + 1
                          )
                          if iw > 0:
                              ih = (
                                  min(anchors[b, k, a, 3], gt_boxes[b][g, 3]) -
                                  max(anchors[b, k, a, 1], gt_boxes[b][g, 1]) + 1
                              )
                              if ih > 0:
                                  ua = float(
                                      (anchors[b, k, a, 2] - anchors[b, k, a, 0] + 1) *
                                      (anchors[b, k, a, 3] - anchors[b, k, a, 1] + 1) +
                                      box_area - iw * ih
                                  )
                                  overlaps[b, k, a, g] = iw * ih / ua
                                  if max_overlap < ((iw * ih / ua)):
                                      max_overlap = iw * ih / ua
                                      max_k = k
                                      max_a = a
            true_index[b, max_k, max_a] = 1


        for k in range(K):
              for a in range(A):
                  if is_inside[b, k, a] == 1:
                      max_overlap = 0
                      max_g = 0
                      for g in range(G):
                          if overlaps[b, k, a, g] > 0:
                              if max_overlap < (overlaps[b, k, a, g]):
                                  max_overlap = overlaps[b, k, a, g]
                                  max_g = g
                      if max_overlap > 0.7:
                          true_index[b, k, a] = 1
                      else:
                          if max_overlap <= 0.3:
                              false_index[b, k, a] = 1

                      if true_index[b, k, a] == 1:
                          ex_width = anchors[b, k, a, 2] - anchors[b, k, a, 0] + 1
                          ex_height = anchors[b, k, a, 3] - anchors[b, k, a, 1] + 1
                          ex_center_x = anchors[b, k, a, 0] + ex_width / 2.0
                          ex_center_y = anchors[b, k, a, 1] + ex_height / 2.0
                          gt_width = gt_boxes[b][max_g, 2] - gt_boxes[b][max_g, 0] + 1
                          gt_height = gt_boxes[b][max_g, 3] - gt_boxes[b][max_g, 1] + 1
                          gt_center_x = gt_boxes[b][max_g, 0] + gt_width / 2.0
                          gt_center_y = gt_boxes[b][max_g, 1] + gt_height / 2.0

                          anchors[b, k, a, 0] = (gt_center_x - ex_center_x) / (ex_width)
                          anchors[b, k, a, 1] = (gt_center_y - ex_center_y) / (ex_height)
                          anchors[b, k, a, 2] = log(gt_width / (ex_width))
                          anchors[b, k, a, 3] = log(gt_height / (ex_height))
    return anchors, true_index, false_index
