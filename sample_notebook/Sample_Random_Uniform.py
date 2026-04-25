import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy import optimize

from sklearn.decomposition import PCA

from mane import mane_2set

np.random.seed(42)

cdata1 = np.random.rand(100,200)
cdata2 = cdata1

data1 = np.random.rand(1000,200)
data2 = np.random.rand(1000,200)


embA, embB, embC = mane_2set(cdata1, None, data1, data2)

plt.figure()
plt.scatter(embA[:,0],embA[:,1], c='y', s=1000)
plt.scatter(embB[:,0],embB[:,1], c='k', s=0.1)
plt.scatter(embC[:,0],embC[:,1], c='r', s=0.1)
