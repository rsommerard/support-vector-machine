import scipy
import numpy as np
import sklearn.datasets as datasets
from matplotlib import pyplot as plt
from matplotlib import lines
from sklearn.svm import SVC

nb_centers = 2
n_samples = 500
X, Y = datasets.make_blobs(centers=nb_centers, n_samples=n_samples, cluster_std=0.3, random_state=0)

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(7, 7)
axes.scatter(X[:,0], X[:,1], c=Y)

clf = SVC(C=0.001, kernel='linear')
clf.fit(X, Y)
hY = clf.predict(X)

print 'Erreur:', abs(hY - Y).sum()

for sv in clf.support_vectors_:
  plt.scatter(sv[0], sv[1], color='g')

w1 = clf.coef_[0][0]
w2 = clf.coef_[0][1]
b = clf.intercept_

print 'w1 = ', w1, '|', 'w2 = ', w2

x11 = 0
x12 = ((-1 * w1 * x11) / w2) - (b / w2)
x21 = 10
x22 = ((-1 * w1 * x21) / w2) - (b / w2)

line = lines.Line2D([x11, x21], [x12, x22], color='k')
plt.axes().add_line(line)

m12 = x12 + (1 / w2)
m22 = x22 + (1 / w2)

line = lines.Line2D([x11, x21], [m12, m22], linestyle='dashed', color='k')
plt.axes().add_line(line)

m12 = x12 - (1 / w2)
m22 = x22 - (1 / w2)

line = lines.Line2D([x11, x21], [m12, m22], linestyle='dashed', color='k')
plt.axes().add_line(line)

plt.show()

################################################################################
################################################################################
################################################################################

X, Y = datasets.make_circles(n_samples=400, factor=.3, noise=.05)

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(7, 7)
axes.scatter(X[:,0], X[:,1], c=Y)

clf = SVC(C=1, kernel='linear')
clf.fit(X, Y)
hY = clf.predict(X)

print 'Erreur:', abs(hY - Y).sum()

plt.show()
