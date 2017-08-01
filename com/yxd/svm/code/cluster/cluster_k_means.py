#coding=utf-8
import ply
from scipy import cluster
from string import center

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
import  sklearn.datasets as ds
from networkx.algorithms.distance_measures import center
from  sklearn.cluster import KMeans
## 设置字符集，防止中文乱码
from sklearn.cluster.tests.test_k_means import n_features
from statsmodels.sandbox.regression.kernridgeregress_class import plt_closeall

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False


#产生数据
N = 1500 #产生1500个点
centers = 4 #4个中心
#参数解释：N=样本数量；n_features=特征数目；center=聚类中心数量；random_state=随机种子
data,y = ds.make_blobs(N,n_features = 2,centers=centers,random_state=170)
#得到的结果是data为1500行2列的数据，y是1500行一列的由0到3组成的数据
#print data.shape
#print y
#构建聚类模型
km = KMeans(n_clusters=centers,random_state=170)
km.fit(data,y)

#输出指标
print "所用样本到中心点的总距离和：",km.inertia_
print "中心点的平均距离：",km.inertia_ / N
print "中心点为：",km.cluster_centers_



#参数解释：cluster_std=聚类方差 数据属性
data2,y2 = ds.make_blobs(N,n_features = 2,centers=centers,cluster_std=(1, 2.5, 0.5,1.5),random_state=170)
#合并两个数据集 按照行累加
data3 = np.vstack((data[y==0][:200],data[y==1][:100],data[y==2][:10],data[y==3][:50]))
#print data3
y3 = np.array([0]*200+[1]*100+[2]*10+[3]*50)
#print y3

#画图参考：http://blog.csdn.net/qiu931110/article/details/68130199
#画图
#预测出值：
y_hat = km.predict(data)
y_hat2 = km.predict(data2)

cm = mpl.colors.ListedColormap(list('rgbm'))
#背景色白色
plt.figure(figsize=(15, 10),facecolor='w')

#原始图
#设置两行四列图形
plt.subplot(321)
#二维平面上画点
xz = data[:,0]
yz = data[:,1]
plt.scatter(xz,yz,c=y,s=30,cmap=cm)#c 代表的是第几种颜色 与cmap参数搭配使用
#获取最大最小值  axis=0按照列；
xx_min, yy_min = np.min(data, axis=0)
xx_max, yy_max = np.max(data, axis=0)
#画图边界
plt.xlim(xx_min,xx_max)
plt.ylim(yy_min,yy_max)
plt.title(u'原始数据')
plt.grid(True)

#画预测图
#字图2
plt.subplot(322)
plt.scatter(xz,yz,c=y_hat,s=30,cmap=cm)#c代表的是形状
plt.xlim(xx_min,xx_max)
plt.ylim(yy_min,yy_max)
plt.title(u'K-Means算法聚类结果')
plt.grid(True)


#字图3 不同方差原始数据
#造数据
xz2 = data2[:,0]
yz2 = data2[:,1]

xx_min2, yy_min2 = np.min(data2, axis=0)
xx_max2, yy_max2 = np.max(data2, axis=0)

plt.subplot(323)
plt.scatter(xz2,yz2,c=y2,s=30,cmap=cm)#c 代表的是第几种颜色 与cmap参数搭配使用
plt.xlim(xx_min2,xx_max2)
plt.ylim(yy_min2,yy_max2)
plt.title(u'不同方差的KMeans原始数据')

plt.subplot(324)
plt.scatter(xz2,yz2,c=y_hat2,s=30,cmap=cm)#c 代表的是第几种颜色 与cmap参数搭配使用
plt.xlim(xx_min2,xx_max2)
plt.ylim(yy_min2,yy_max2)
plt.title(u'不同方差的KMeans原始数据')

plt.tight_layout()
plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)
plt.show()


