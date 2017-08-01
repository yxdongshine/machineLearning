#coding=utf-8
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as mpy
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#定义特定中心点的数据
center_data = [[1,1],
               [-1,1],
               [1,-1]
               ]
center_len = len(center_data)
N = 1500
xs,y = make_blobs(N,centers=center_data)
#print xs.shape,y

#开始训练
km = KMeans(n_clusters=center_len)
#查看时间差
kmt = time.time()
km.fit(xs)
total_time = time.time - kmt
print "km时间：%.4fs" % total_time
#输出指标
print "所用样本到中心点的总距离和：",km.inertia_
print "中心点的平均距离：",km.inertia_ / N
print "中心点为：",km.cluster_centers_

print"==================================================="

#训练batchKMeans
bs = 100
km_b = MiniBatchKMeans(n_clusters=center_len,batch_size=bs)

#查看时间差
kmbt = time.time();
km_b.fit(xs)
total_time = time.time - kmt
print "kmbt时间：%.4fs" % total_time
#输出指标
print "所用样本到中心点的总距离和：",km_b.inertia_
print "中心点的平均距离：",km_b.inertia_ / N
print "中心点为：",km_b.cluster_centers_