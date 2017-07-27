#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import  svm
import matplotlib.colors
import matplotlib as mpl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False


#这里生成1100行2列数组 目标值 前1000行结果-1 后一百为1的数据集
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
x = np.r_[
    rng.randn(n_samples_1, 2),
    rng.randn(n_samples_2, 2)

]
y = [-1] * (n_samples_1) + [1] * (n_samples_2)
print x.shape,y.__len__()

#这里调用不同的核函数和参数比较优良
svmkfp = [
           svm.SVC(C=1, kernel='linear'),
           svm.SVC(C=100, kernel='linear'),
           svm.SVC(C=100, kernel='rbf', gamma=10, class_weight={-1: 1, 1: 0.5}),#类别的权重，字典形式传递。设置第几类的参数C为weight*C
           svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={-1: 1, 1: 10})
]
#图形标题
titles = 'kernel=linear', 'kernel=linearandWeight=100', 'kernel=rbf, W=0.5', 'kernel=rbf, W=10'

x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # x0列范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # x1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点

print x1.flat
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 生成一行N列的数据测试

#颜色区分
cm_light = matplotlib.colors.ListedColormap(['#77E0A0', '#FF8080'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r'])

#训练比较指标得分
for i, svmkfp in enumerate(svmkfp):
    svmkfp.fit(x, y)

    y_hat = svmkfp.predict(x)

    print i+1, 'times：'
    print '准确率：\t', accuracy_score(y, y_hat)
    print '正确率 ：\t', precision_score(y, y_hat, pos_label=1)
    print '召回率：\t', recall_score(y, y_hat, pos_label=1)
    print 'F值  ：\t', f1_score(y, y_hat, pos_label=1)


