cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

def bbox_to_label(
        np.ndarray[DTYPE_t, ndim=4] anchors,
        np.ndarray[DTYPE_int_t, ndim=3] true_index,
        np.ndarray[DTYPE_t, ndim=3] false_index):
    """
    Parameters
    ----------
    anchors: (Batch_Size, K, A, 4) ndarray of float
    is_inside: (Batch_Size, K, A) ndarray of int
    gt_boxes: (Batch, G, 4) ndarray of float
    Returns
    -------
    overlaps: (Batch_Size, K, A, G) ndarray of overlap between anchors and gt_boxes
    """
    cdef unsigned int Batch_Size = anchors[0]
    cdef unsigned int K = anchors.shape[1]
    cdef unsigned int A = anchors.shape[2]
    cdef unsigned int G = gt_boxes.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=4] overlaps = np.zeros((Batch_Size, K, A, G), dtype=DTYPE)
    cdef np.ndarray[DTYPE_int_t, ndim=3] true_index = np.zeros((Batch_Size, K, A), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=3] false_index = np.zeros((Batch_Size, K, A), dtype=DTYPE_int)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef DTYPE_t max_overlap
    cdef unsigned int k, a, b, g, max_k, max_a

    for b in range(Batch_Size):
        for g in range(G):
            box_area = (
                (gt_boxes[b, g, 2] - gt_boxes[b, g, 0] + 1) *
                (gt_boxes[b, g, 3] - gt_boxes[b, g, 1] + 1)
            )
            max_overlap = 0
            for k in range(K):
                  for a in range(A):
                      if is_inside[b, k, a] == 1:
                          iw = (
                              min(anchors[b, k, a, 2], gt_boxes[b, g, 2]) -
                              max(anchors[b, k, a, 0], gt_boxes[b, g, 0]) + 1
                          )
                          if iw > 0:
                              ih = (
                                  min(anchors[b, k, a, 3], gt_boxes[b, g, 3]) -
                                  max(anchors[b, k, a, 1], gt_boxes[b, g, 1]) + 1
                              )
                              if ih > 0:
                                  ua = float(
                                      (anchors[b, k, a, 2] - anchors[b, k, a, 0] + 1) *
                                      (anchors[b, k, a, 3] - anchors[b, k, a, 1] + 1) +
                                      box_area - iw * ih
                                  )
                                  overlaps[b, k, a, g] = iw * ih / ua
                                  if max_overlap < (iw * ih / ua):
                                      max_overlap = iw * ih / ua
                                      max_k = k
                                      max_a = a
                  true_index[b, max_k, max_a] = 1
                  anchors[b, max_k, max_a] = 

        for k in range(K):
              for a in range(A):
                  if is_inside[b, k, a] == 1:
                      max_overlap = 0
                      for g in range(G):
                          if overlaps[b, k, a, g] > 0:
                              if max_overlap < overlaps[b, k, a, g]:
                                  max_overlap = overlaps[b, k, a, g]
                      if max_overlap > 0.7:
                          true_index[b, k, a] = 1
                      else:
                          if max_overlap <= 0.3:
                              false_index[b, k, a] = 1
    return overlaps, true_index, false_index
