# encoding=utf-8

import scipy
import numpy as np
import sklearn.datasets as datasets
import sklearn.cross_validation as cv
from matplotlib import pyplot as plt
from matplotlib import lines
from sklearn.svm import SVC

nb_centers = 2
n_samples = 500
data = datasets.make_blobs(centers=nb_centers, n_samples=n_samples, cluster_std=0.3, random_state=0)
X, Y = data

X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size=0.3, train_size=0.7)

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(7, 7)
axes.scatter(X[:,0], X[:,1], c=Y)

clf = SVC(C=0.001, kernel='linear')
clf.fit(X_train, Y_train)
hY = clf.predict(X_test)

print '-' * 80
print 'Linear data and linear SVC'
print 'Erreur:', abs(hY - Y_test).sum()
print 'Les prédictions sont parfaites. On obtient des erreurs de 0.'
print '-' * 80

for sv in clf.support_vectors_:
  plt.scatter(sv[0], sv[1], color='g')

w1 = clf.coef_[0][0]
w2 = clf.coef_[0][1]
b = clf.intercept_

print 'w1 = ', w1, '|', 'w2 = ', w2
print '-' * 80

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

data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
X, Y = data
X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size=0.3, train_size=0.7)

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(7, 7)
axes.scatter(X[:,0], X[:,1], c=Y)
plt.show()

Xb_train = []
for x in X_train:
    Xb_train.append([1, x[0], x[1], x[0]**2, x[1]**2])

Xb_test = []
for x in X_test:
    Xb_test.append([1, x[0], x[1], x[0]**2, x[1]**2])

clf = SVC(kernel='linear')
clf.fit(Xb_train, Y_train)
hY = clf.predict(Xb_test)

print 'No linear data and linear SVC'
print 'Erreur:', abs(hY - Y_test).sum()
print 'Les prédictions sont parfaites. On obtient des erreurs de 0.'
print '-' * 80

################################################################################
################################################################################
################################################################################

data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
X, Y = data
X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size=0.3, train_size=0.7)

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(7, 7)
axes.scatter(X[:,0], X[:,1], c=Y)
plt.show()

clf = SVC(kernel='poly', degree=2)
clf.fit(X_train, Y_train)
hY = clf.predict(X_test)

print 'No linear data and poly SVC'
print 'Erreur:', abs(hY - Y_test).sum()
print 'Les prédictions sont parfaites. On obtient des erreurs de 0.'
print '-' * 80

################################################################################
################################################################################
################################################################################

data = datasets.make_circles(n_samples=n_samples, factor=0.9, noise=.05)
X, Y = data
X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size=0.3, train_size=0.7)

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(7, 7)
axes.scatter(X[:,0], X[:,1], c=Y)
plt.show()

clf = SVC(kernel='poly', degree=2)
clf.fit(X_train, Y_train)
hY = clf.predict(X_test)

print 'No linear data and poly SVC'
print 'Erreur:', abs(hY - Y_test).sum(), 'sur', len(Y_test)
print 'Les prédictions moins bonnes car les données sont moins séparables. On obtient plus d\'erreurs.'
print '-' * 80
