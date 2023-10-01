import warnings
import numpy as np
from scipy.spatial.distance import euclidean, cosine, jaccard

warnings.filterwarnings('ignore')

# Compute correlation
def corr(x, y):
	xbar = np.mean(x)
	ybar = np.mean(y)
	cov_xy = np.sum((x - xbar) * (y - ybar))
	var_x = np.sqrt(np.sum((x - xbar) ** 2))
	var_y = np.sqrt(np.sum((y - ybar) ** 2))
	return cov_xy / (var_x * var_y)


# (a)
a = {
	 'x1' : np.array([1, 1, 1, 1]), 
	 'x2' : np.array([2, 2, 2, 2])
}

# (b)
b = {
	 'x1' : np.array([0, 1, 0, 1]),
	 'x2' : np.array([1, 0, 1, 0])
}

# (c)
c = {
	 'x1' : np.array([0, -1, 0, 1]),
	 'x2' : np.array([1, 0, -1, 0])
}

# Compute the distances
## For (a)
print('\nDistance computation for (a) : ')
print(' - Cosine(x1, x2) = ', cosine(a['x1'], a['x2']))
print(' - Correlation(x1, x2) = ', corr(a['x1'], a['x2'])) # Will throw error due to division by zero
print(' - Euclidean(x1, x2) = ', euclidean(a['x1'], a['x2']))

## For (b)
print('\nDistance computation for (b) : ')
print(' - Cosine(x1, x2) = ', cosine(b['x1'], b['x2']))
print(' - Correlation(x1, x2) = ', corr(b['x1'], b['x2']))
print(' - Euclidean(x1, x2) = ', euclidean(b['x1'], b['x2']))
print(' - Jaccard(x1, x2) = ', jaccard(b['x1'], b['x2']))

## For (c)
print('\nDistance computation for (c) : ')
print(' - Cosine(x1, x2) = ', cosine(c['x1'], c['x2']))
print(' - Correlation(x1, x2) = ', corr(c['x1'], c['x2']))
print(' - Euclidean(x1, x2) = ', euclidean(c['x1'], c['x2']))