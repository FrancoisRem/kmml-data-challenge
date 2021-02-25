from unittest import TestCase

import numpy as np
from numpy import testing

from utils import binary_regression_labels


class Test(TestCase):
    def test_binary_regression_labels(self):
        testing.assert_equal(
            binary_regression_labels(np.array([0, 0, 1])),
            np.array([-1, -1, 1]))
