from unittest import TestCase

from numpy import testing

from kernels import *


class Test(TestCase):
    def test_linear_kernel_gram_matrix_1d(self):
        X = np.vstack([2])
        Y = np.vstack([3])
        testing.assert_almost_equal(linear_kernel_gram_matrix(X, Y), 6)

        X = np.vstack([7])
        Y = np.vstack([0])
        testing.assert_almost_equal(linear_kernel_gram_matrix(X, Y), 0)

    def test_linear_kernel_gram_matrix_2d(self):
        X = np.array([[1, 2], [6, 6]])
        Y = np.array([[0, 1], [2, 1]])
        expected = np.array([[2, 4], [6, 18]])
        testing.assert_almost_equal(linear_kernel_gram_matrix(X, Y), expected)

        X = np.array([[1, 0]])
        Y = np.array([[0, 1], [2, 1]])
        expected = np.array([[0, 2]])
        testing.assert_almost_equal(linear_kernel_gram_matrix(X, Y), expected)

    def test_gaussian_kernel_gram_matrix_1d(self):
        X = np.vstack([2])
        Y = np.vstack([3])
        # expected: exp(-1*||3-2||^2)=exp(-1)~0.3678794411
        testing.assert_almost_equal(gaussian_kernel_gram_matrix(X, Y, gamma=1),
                                    0.3678794411)

        testing.assert_almost_equal(
            gaussian_kernel_gram_matrix(X, Y, gamma='auto'),
            0.3678794411)

        X = np.vstack([7])
        Y = np.vstack([0])
        # expected: exp(-0.05*||7-0||^2)=exp(-2.45)~0.08629358649
        testing.assert_almost_equal(
            gaussian_kernel_gram_matrix(X, Y, gamma=0.05),
            0.08629358649)

    def test_gaussian_kernel_gram_matrix_2d(self):
        X = np.array([[1, 2], [6, 6]])
        Y = np.array([[0, 1], [2, 1]])

        # expected = np.array([[exp(-2), exp(-2)], [exp(-61), exp(-41)]])
        expected = np.array(
            [[0.13533528323, 0.13533528323], [3.2213403e-27, 1.5628822e-18]])
        testing.assert_almost_equal(gaussian_kernel_gram_matrix(X, Y, gamma=1),
                                    expected)

        # expected = np.array([[exp(-1), exp(-1)], [exp(-30.5), exp(-20.5)]])
        expected = np.array([[3.67879441e-01, 3.67879441e-01],
                             [5.67568523e-14, 1.25015287e-09]])
        testing.assert_almost_equal(
            gaussian_kernel_gram_matrix(X, Y, gamma='auto'),
            expected)

        X = np.array([[1, 0]])
        Y = np.array([[0, 1], [2, 1]])
        # expected = [[exp(-2), exp(-2)]]
        expected = np.array([[0.13533528323, 0.13533528323]])
        testing.assert_almost_equal(gaussian_kernel_gram_matrix(X, Y, gamma=1),
                                    expected)

    def test_cosine_similarity_kernel_gram_matrix(self):
        X = np.vstack([2])
        Y = np.vstack([3])
        testing.assert_almost_equal(cosine_similarity_kernel_gram_matrix(X, X),
                                    1)
        testing.assert_almost_equal(cosine_similarity_kernel_gram_matrix(X, Y),
                                    1)

        X = np.array([[1, 2], [6, 6]])
        Y = np.array([[0, 1], [2, 1]])
        expected = np.array([[1., 0.9486833],
                             [0.9486833, 1.]])
        testing.assert_almost_equal(cosine_similarity_kernel_gram_matrix(X, X),
                                    expected)
        expected = np.array([[0.89442719, 0.8],
                             [0.70710678, 0.9486833]])
        testing.assert_almost_equal(cosine_similarity_kernel_gram_matrix(X, Y),
                                    expected)
