import sys
import traceback
import os

import numpy as np
import numpy.testing as nptest
import warnings

from wquantiles import quantile_1d, quantile


class TestPercentiles:
    def setup_method(self, method):
        # Sorted 1D array
        self.a1d = np.array([0, 10, 20, 25, 30, 30, 35, 50])
        self.a1d_w = np.array([0, 1, 0, 1, 2, 2, 2, 1])
        # Unsorted 1D array
        self.a1du = np.array([30, 25, 0, 50, 30, 20, 35, 10])
        self.a1du_w = np.array([2, 1, 0, 1, 2, 0, 2, 1])
        # 2D array
        self.a2d = np.array([[1, 3, 5, 6, 8,],
                             [5, 3, 4, 10, 3],
                             [4, 1, 1, 5, 7],
                             [10, 3, 5, 7, 8],
                             [1, 1, 6, 4, 6]])
        self.aw = np.array([0, 1, 0, 1, 2])
        self.layers = [np.array([[70, 51, 20, 19, 58],
                               [20, 50,  3, 28,  4],
                               [87, 71, 72, 40,  9],
                               [49, 48, 78, 72, 83],
                               [48, 96, 80, 75, 77]]),
                         np.array([[18, 43, 33, 36, 62],
                               [62, 71, 40, 29, 26],
                               [80, 51, 23, 33, 89],
                               [20, 92, 30, 84, 63],
                               [12, 31, 64, 88, 55]]),
                         np.array([[29, 91, 66, 42, 86],
                               [ 2, 92, 64, 67, 12],
                               [23, 33, 59,  9, 13],
                               [ 4, 81, 69, 22, 79],
                               [ 6, 73,  2, 37, 25]]),
                         np.array([[32, 51, 41, 63, 29],
                               [35, 93, 67,  1, 29],
                               [25, 15, 72, 88, 27],
                               [99, 16, 23, 61, 82],
                               [52, 82, 32, 72, 74]]),
                         np.array([[67, 92, 35, 50, 11],
                               [22, 89, 93,  6, 87],
                               [49, 62,  3, 92, 25],
                               [63, 34, 52, 51, 82],
                               [34, 32, 23, 23, 44]])]
        self.a3d = np.empty((5,5,5))
        for i, layer in enumerate(self.layers):
            self.a3d[:, :, i] = layer

    def test_median(self):
        assert np.median(self.a1D) == 27.5
        assert np.median(self.a2D) == 5.0
        assert list(np.median(self.a2d, axis=-1)) == [ 5.,  4.,  4.,  7.,  4.]
        assert list(np.median(self.a2d, axis=1)) == [ 5.,  4.,  4.,  7.,  4.]
        assert list(np.median(self.a2d, axis=0)) == [ 4.,  3.,  5.,  6.,  7.]

    def test_weighted_median_1d_sorted(self):
        # Median of the sorted array
        assert quantile_1d(self.a1d, self.a1d_w, 0.5), 30
        assert quantile_1d(self.a1d, np.ones_like(self.a1d), 0.5), 27.5

    def test_weighted_median_1d_unsorted(self):
        # Median of the unsorted array
        assert quantile_1d(self.a1du, self.a1du_w, 0.5) == 30
        assert quantile_1d(self.a1du, np.ones_like(self.a1du), 0.5) == 27.5

    def test_weighted_median_3d(self):
        arr1 = quantile(self.a3d, self.aw, 0.5)
        arr2 = np.array([
            [43.66666667, 91., 35., 50., 23.],
            [30.66666667, 89., 75.66666667, 6., 48.33333333],
            [49., 54.66666667, 16.33333333, 89.33333333, 26.33333333],
            [63., 34., 37.33333333, 57.66666667, 82.],
            [34., 32., 29., 37., 51.33333333]
        ])
        # print(arr1, arr2)
        nptest.assert_array_almost_equal(arr1, arr2)
