import os
import pandas as pd
import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, int32_t, int64_t
#from AIList_core cimport AIList, ailist_t, ailist_init


def get_include():
    """
    Get file directory if C headers
    
    Arguments:
    ---------
        None
    Returns:
    ---------
        str (Directory to header files)
    """

    return os.path.split(os.path.realpath(__file__))[0]


cdef aiarray_t *_offline_bcpseg(const double[::1] values, double truncate, double cutoff):
    # Find length of values
    cdef int length = values.size

    # Segment
    cdef aiarray_t *c_segments = offline_bcp_segment(&values[0], length, truncate, cutoff)

    return c_segments

cdef void _offline_bcpseg_labeled(const double[::1] values, labeled_aiarray_t *c_segments, char *label, double truncate, double cutoff):
    # Find length of values
    cdef int length = values.size

    # Segment
    offline_bcp_segment_labeled(&values[0], c_segments, label, length, truncate, cutoff)


cdef aiarray_t *_online_bcpseg(const double[::1] values, double cutoff, double hazard):
    # Find length of values
    cdef int length = values.size

    # Segment
    cdef aiarray_t *c_segments = online_bcp_segment(&values[0], length, cutoff, hazard)

    return c_segments

cdef void _online_bcpseg_labeled(const double[::1] values, labeled_aiarray_t *c_segments, char *label, double cutoff, double hazard):
    # Find length of values
    cdef int length = values.size

    # Segment
    online_bcp_segment_labeled(&values[0], c_segments, label, length, cutoff, hazard)


cdef aiarray_t *_online_bcpseg_both(const double[::1] forward_values, const double[::1] reverse_values,
                                   double cutoff, double hazard, int offset):
    # Find length of values
    cdef int length = forward_values.size

    # Segment
    cdef aiarray_t * c_segments = online_bcp_both(&forward_values[0], &reverse_values[0], length, cutoff, hazard, offset)

    return c_segments

cdef void _online_bcpseg_both_labeled(const double[::1] forward_values, const double[::1] reverse_values,
                                   labeled_aiarray_t *c_segments, char *label, double cutoff, double hazard, int offset):
    # Find length of values
    cdef int length = forward_values.size

    # Segment
    online_bcp_both_labeled(&forward_values[0], &reverse_values[0], c_segments, label, length, cutoff, hazard, offset)


def bcpseg(np.ndarray values, np.ndarray labels=None, double truncate=-100, double cutoff=0.75, str method="online",
           double hazard=100, int offset=10):
    """
    Implementation of a Bayesian Change Point Detection
    algorithm to segment values into Intervals

    Parameters
    ----------
        values : numpy.ndarray
			Floats to segment
        labels : numpy.ndarray
            Labels for breakup segmentation
        truncate : float
            Tolerance during offline segmentation [default=-100]
        cutoff : float
            Probability threshold for determining segment bounds [default=0.75]
        method : str
            Method to use: offline, online, online_both [default:'online']
        hazard : float
            Expected typical segment length [default:100]
        offset : int
            Number to skip before calculating probability [default:10]

    Returns
    -------
        segments : AIList
			Segment intervals
    """

    # Initilaize segments
    cdef IntervalArray segments
    cdef aiarray_t *c_segments
    cdef LabeledIntervalArray lsegments
    cdef labeled_aiarray_t *c_lsegments
    cdef const double[::1] labeled_values
    cdef np.ndarray unique_labels
    cdef char *label

    # Determine method
    if method == "online":
        if labels is None:
            c_segments = _online_bcpseg(values, cutoff, hazard)
        else:
            # Find unique labels
            labels = labels.astype(bytes)
            unique_labels = pd.unique(labels)
            c_lsegments = labeled_aiarray_init()
            for label in unique_labels:
                label_values = values[labels==label]
                _online_bcpseg_labeled(label_values, c_lsegments, label, cutoff, hazard)
    
    elif method == "online_both":
        if labels is None:
            reverse_values = np.ascontiguousarray(np.flip(values))
            c_segments = _online_bcpseg_both(values, reverse_values, cutoff, hazard, offset)
        else:
            # Find unique labels
            labels = labels.astype(bytes)
            unique_labels = pd.unique(labels)
            c_lsegments = labeled_aiarray_init()
            for label in unique_labels:
                label_values = values[labels==label]
                reverse_values = np.ascontiguousarray(np.flip(label_values))
                _online_bcpseg_both_labeled(label_values, reverse_values, c_lsegments, label, cutoff, hazard, offset)
    else:
        if labels is None:
            c_segments = _offline_bcpseg(values, truncate, cutoff)
        else:
            # Find unique labels
            labels = labels.astype(bytes)
            unique_labels = pd.unique(labels)
            c_lsegments = labeled_aiarray_init()
            for label in unique_labels:
                label_values = values[labels==label]
                _offline_bcpseg_labeled(label_values, c_lsegments, label, truncate, cutoff)

    # Wrap segments
    if labels is None:
        segments = IntervalArray()
        segments.set_list(c_segments)

        return segments
    else:
        lsegments = LabeledIntervalArray()
        lsegments.set_list(c_lsegments)

        return lsegments
