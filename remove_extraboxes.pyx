cimport cython
import numpy as np
cimport numpy as np

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

def remove_extraboxes(
        np.ndarray[DTYPE_int_t, ndim=1] array1,
        np.ndarray[DTYPE_int_t, ndim=1] array2,
        np.ndarray[DTYPE_int_t, ndim=1] select,
        np.ndarray[DTYPE_int_t, ndim=1] batch):
    """
    Parameters
    ----------
    array1: (A) ndarray of int
    array2: (A) ndarray of int
    select: (B) ndarray of int
    Returns
    -------
    extract_array1 : (64) ndarray of index of remove boxes
    extract_array2 : (64) ndarray of index of remove boxes
    """
    cdef unsigned int remove_size = select.shape[0]
    cdef np.ndarray[DTYPE_int_t, ndim=1] extract_array1 = np.zeros((remove_size), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=1] extract_array2 = np.zeros((remove_size), dtype=DTYPE_int)
    cdef unsigned int rs

    for rs in range(remove_size):
        extract_array1[rs] = array1[select[rs]]
        extract_array2[rs] = array2[select[rs]]
    return batch, extract_array1, extract_array2
