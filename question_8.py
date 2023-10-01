# Import packages.
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Initialize data
n = 6
X = np.array([
    [3, 3], [3, 4], [5, 4],
    [1, 1], [2, 2], [4, 1]
]).astype(float)
y = np.array([1, 1, 1, -1, -1, -1]).astype(float)

# QP problem
m, n = X.shape

## 1. Objective function $\frac{1}{2}\alpha^TP\alpha + q^T\alpha$
K = np.matmul(X, X.T)
P = matrix(y.reshape(-1, 1) @ y.reshape(-1, 1).T * K)
q = matrix(np.ones((m, 1)) * -1)

## 2. Equality constraint $A\alpha \eq b$
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))          

## 3. Inequality constraint $G\alpha \le h$
G = matrix(np.eye(m) * -1)
h = matrix(np.zeros(m))

# Solve the QP
solution = solvers.qp(P, q, G, h, A, b)
alphas = np.array(solution['x'])
print('Solution for alpha : \n', alphas)

# Find the support vectors
support_vectors_ind = (alphas > 1e-4).flatten()
X_sv = X[support_vectors_ind]
y_sv = y[support_vectors_ind]
alphas_sv = alphas[support_vectors_ind]

# Calculate W and b
W = np.sum(X * (alphas.flatten() * y).reshape(-1, 1), axis=0)
b = y_sv - X_sv @ W
b = np.sum(b) / b.size

# Print the coefficients
print('W = ', W)
print('b = ', b)

# Plot the results
## 1. Plot positive class
plt.scatter(X[:3][:, 0], X[:3][:, 1], color='tab:red', marker='o', label='Positive class')
plt.scatter(X[3:][:, 0], X[3:][:, 1], color='tab:blue', marker='v', label='Negative class')

# Plot the line
x1 = np.linspace(0, 5, num=100)
x2 = (-b - W[0] * x1) / W[1]
plt.plot(x1, x2)
plt.legend()
plt.show()