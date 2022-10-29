# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""

from mjc import mjc
import numpy as np
import unittest


def dummy_data(phase=0, with_time=False, time_offset: float = 0):
    cos_base = np.array(np.cos(np.linspace(0, 4 * np.pi + phase, 100)))
    amplitude = np.linspace(0, 4 * np.pi, 100)
    noise = np.random.uniform(-0.1, 0.1, 100)
    data = amplitude*cos_base + noise
    if with_time:
        ret = np.vstack((np.linspace(0, 1, 100) + time_offset, data))
    else:
        ret = data
    return ret

class mjcTester(unittest.TestCase):
    def test_ndarray_notime(self):
        s1 = dummy_data()
        s2 = dummy_data(np.pi/4)
        mjc(s1, s2)

    def test_ndarray_withtime(self):
        s1 = dummy_data(with_time=True)
        s2 = dummy_data(with_time=True, time_offset=0.2)
        mjc(s1, s2)

    def test_list_notime(self):
        s1 = list(dummy_data())
        s2 = list(dummy_data(np.pi/4))
        mjc(s1, s2)

    def test_list_withtime(self):
        s1 = list(dummy_data(with_time=True))
        s2 = list(dummy_data(with_time=True, time_offset=0.2))
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

    def test_plot(self):
        s1 = dummy_data(with_time=True)
        s2 = dummy_data(with_time=True, time_offset=0)
        s2[1] += 2
        mjc(s1, s2, show_plot=True)

if __name__ == '__main__':
    unittest.main()