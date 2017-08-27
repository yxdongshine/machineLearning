#coding=utf-8
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import metrics
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
total_time = time.time() - kmt
print "km时间" + total_time
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
total_time = time.time() - kmbt
print "kmbt时间：%.4fs" % total_time
#输出指标
print "所用样本到中心点的总距离和：",km_b.inertia_
print "中心点的平均距离：",km_b.inertia_ / N
print "中心点为：",km_b.cluster_centers_

#预算
y_hat_km = km.predict(xs)
y_hat_kmb = km_b.predict(xs)


#指标评优
score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    metrics.adjusted_mutual_info_score,
    metrics.mutual_info_score,
]

for score_func in score_funcs:
    t0 = time.time()
    km_scores = score_func(y, y_hat_km)
    print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__, km_scores, time.time() - t0))

    t0 = time.time()
    mbkm_scores = score_func(y, y_hat_kmb)
    print(
    "Mini Batch K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs\n" % (score_func.__name__, mbkm_scores, time.time() - t0))

#画图比较
#画出两行两列
cm = mpl.colors.ListedColormap(list('rgb'))
#背景色白色
plt.figure(figsize=(15, 10),facecolor='w')
plt.subplot(221)
#二维平面上画点
xz = xs[:,0]
yz = xs[:,1]
plt.scatter(xz,yz,c=y,s=30,cmap=cm)#c 代表的是第几种颜色 与cmap参数搭配使用
#获取最大最小值  axis=0按照列；
xx_min, yy_min = np.min(xs, axis=0)
xx_max, yy_max = np.max(xs, axis=0)
#画图边界
plt.xlim(xx_min,xx_max)
plt.ylim(yy_min,yy_max)
plt.title(u'原始数据')
plt.grid(True)

#画km的预测值
plt.subplot(222)
#二维平面上画点
xz = xs[:,0]
yz = xs[:,1]
plt.scatter(xz,yz,c=y_hat_km,s=30,cmap=cm)#c 代表的是第几种颜色 与cmap参数搭配使用
#获取最大最小值  axis=0按照列；
xx_min, yy_min = np.min(xs, axis=0)
xx_max, yy_max = np.max(xs, axis=0)
#画图边界
plt.xlim(xx_min,xx_max)
plt.ylim(yy_min,yy_max)
plt.title(u'km的预测值')
plt.grid(True)
#画kmb的预测值
plt.subplot(223)
#二维平面上画点
xz = xs[:,0]
yz = xs[:,1]
plt.scatter(xz,yz,c=y_hat_kmb,s=30,cmap=cm)#c 代表的是第几种颜色 与cmap参数搭配使用
#获取最大最小值  axis=0按照列；
xx_min, yy_min = np.min(xs, axis=0)
xx_max, yy_max = np.max(xs, axis=0)
#画图边界
plt.xlim(xx_min,xx_max)
plt.ylim(yy_min,yy_max)
plt.title(u'kmb的预测值')
plt.grid(True)

plt.tight_layout()
plt.suptitle(u'KMeans与BatchKMeans比较', fontsize=18)
plt.show()