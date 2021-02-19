import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

x = np.linspace (0, 100, 200)
y1 = stats.gamma.pdf(x, a=4.8, scale=2.3)


X = [[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10 , 11, 12]]


Xs = [np.sum(i) for i in X]

print(Xs)

print(Xs.index(max(Xs)))
