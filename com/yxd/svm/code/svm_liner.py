#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets , svm
import matplotlib as mpl

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

iris = datasets.load_iris()
X = iris.data
y = iris.target

#取出前面两列数据
X = X[y != 0, :2]
border = len(X )* 0.8

#按照28比例分开训练测试
X_train = X[:border]
X_test = X[border:]
y_train = y[:border]
y_test = y[border:100]

print  len(X_test)
print len(y_test)

#开始使用svm 线性核函数训练
svm_mode = svm.SVC(kernel = 'linear')
svm_mode.fit(X_train,y_train)
score = svm_mode.score(X_test,y_test)
print score

#预测
y_predict = svm_mode.predict(X_test)
print  y_predict

## 7. 画图
plt.figure(figsize=(12,6), facecolor='w')
ln_x_test = range(len(X_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'实际值')
plt.plot(ln_x_test, y_predict, 'b-', lw=2, label=u'预测值，$R^2$=%.3f' % score)
plt.xlabel(u'测试值X')
plt.ylabel(u'目标值y')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.title(u'svm 线性模式结果')
plt.show()
