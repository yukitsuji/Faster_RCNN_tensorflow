cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def inside_image(
        np.ndarray[DTYPE_t, ndim=3] boxes,
        unsigned int width,
        unsigned int height):
    """
    Parameters
    ----------
    boxes: (K, A, 4) ndarray of float
    width: width of input images
    height: height of input images
    Returns
    -------
    is_inside: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int K = boxes.shape[0]
    cdef unsigned int A = boxes.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] is_inside = np.zeros((K, A), dtype=DTYPE)
    cdef unsigned int k, a
    for k in range(K):
        for a in range(A):
            if boxes[k, a, 0] >= 0:
                if boxes[k, a, 1] >= 0:
                    if boxes[k, a, 2] < width:
                        if boxes[k, a, 3] < height:
                            is_inside[k, a] = 1
    return is_inside
