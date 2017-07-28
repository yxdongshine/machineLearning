#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import  svm
import matplotlib.colors
import matplotlib as mpl

## 设置字符集，防止中文乱码
from sympy.core.tests.test_arit import same_and_same_prec

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False


np.random.seed(0)
X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
y = [1] * 10 + [-1] * 10
print X.shape, y.__len__()
print X
print ("----------------------")
print y

#随机权重
sample_weight_last_ten = abs(np.random.randn(len(X)))
sample_weight_constant = np.ones(len(X))
print  sample_weight_last_ten
print ("----------------------")
print  sample_weight_constant
# and bigger weights to some outliers
sample_weight_last_ten[15:] *= 5
sample_weight_last_ten[9] *= 15

print ("-----------big w-----------")
print  sample_weight_last_ten
print ("----------------------")
print  sample_weight_constant


#开始训练mode
# fit the model

clf_weights = svm.SVC()
clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)

clf_no_weights = svm.SVC()
clf_no_weights.fit(X, y)


#定义画图 函数
def plot_decision_function(classifier, sample_weight, axis, title):
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cm_light = matplotlib.colors.ListedColormap(['#77E0A0', 'y'])
    cm_dark = matplotlib.colors.ListedColormap(['g', 'r'])
    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.1, cmap=cm_light)
    axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
                 cmap=cm_dark)

    axis.axis('off')
    axis.set_title(title)

#生成数据测试并且画图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
                       "Constant weights")
#plot_decision_function(clf_weights, sample_weight_last_ten, axes[1],
#                     "Modified weights")

plt.show()


