# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""

from src.mjc import mjc
import numpy as np
import unittest


def dummy_data(phase=0, with_time=False, fs: int = 100, offset=0):
    cos_base = np.array(np.cos(np.linspace(0, 4 * np.pi + phase, fs)))
    amplitude = np.linspace(0, 4 * np.pi, fs)
    noise = np.random.uniform(-0.1, 0.1, fs)
    data = amplitude * cos_base + noise + offset
    if with_time:
        ret = np.vstack((np.linspace(0, 1, fs), data))
    else:
        ret = data
    return ret


class mjcTester(unittest.TestCase):
    def test_ndarray_notime(self):
        s1 = dummy_data()
        s2 = dummy_data(np.pi / 4)
        mjc(s1, s2)

    def test_ndarray_withtime(self):
        s1 = dummy_data(with_time=True)
        s2 = dummy_data(with_time=True, offset=2)
        mjc(s1, s2)

    def test_list_notime(self):
        s1 = list(dummy_data())
        s2 = list(dummy_data(np.pi / 4))
        mjc(s1, s2)

    def test_list_withtime(self):
        s1 = list(dummy_data(with_time=True))
        s2 = list(dummy_data(with_time=True, offset=2))
        mjc(s1, s2)

    def test_mismatched_dimensions(self):
        s1 = dummy_data()
        s2 = dummy_data(with_time=True)
        self.assertRaises(ValueError, mjc, s1, s2)

    def test_incorrect_dimensions(self):
        s1 = np.empty(shape=[1, 1, 1])
        s2 = dummy_data()
        self.assertRaises(AssertionError, mjc, s1, s2)

    def test_nonnumeric(self):
        s1 = dummy_data().astype(np.bool_)
        s2 = dummy_data().astype(np.bool_)
        self.assertRaises(AssertionError, mjc, s1, s2)

    def test_overlapping(self):
        s1 = dummy_data(with_time=True)[:, :-30]
        s2 = dummy_data(offset=2, with_time=True)[:, 30:]
        mjc(s1, s2)

    def test_different_sampling_rate(self):
        s1 = dummy_data(with_time=True)
        s2 = dummy_data(offset=2, fs=30, with_time=True)
        mjc(s1, s2)

    def test_plot(self):
        s1 = dummy_data(with_time=True)[:, 30:]
        s2 = dummy_data(offset=2, fs=30, with_time=True)[:, :-10]
        mjc(s1, s2, show_plot=True)


if __name__ == '__main__':
    unittest.main()
