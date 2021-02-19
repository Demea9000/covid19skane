import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

x = np.linspace (0, 100, 200)
y1 = stats.gamma.pdf(x, a=4.8, scale=2.3)
plt.plot(x, y1)
plt.show()
