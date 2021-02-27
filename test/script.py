# Untested scripts used to compute magic numbers for the tests in this folder.

import cvxpy as cp
import numpy as np

n = 2
X = np.array([[1, 2], [6, 6]])
y = 2 * np.array([0, 1]) - 1
K = X @ X.T
alpha = 1
print(K)
print(sorted(np.linalg.eig(K)[0]))

# n = 1
# X = np.vstack([1])
# y = 2 * np.array([1]) - 1
# K = X @ X.T
# alpha = 1
# print(K)

coef = cp.Variable(n)
problem = cp.Problem(cp.Minimize(
    1 / n * cp.sum(
        cp.logistic(-cp.multiply(K @ coef, y))) + alpha * cp.quad_form(
        coef, K)))
problem.solve()
print(coef.value)
