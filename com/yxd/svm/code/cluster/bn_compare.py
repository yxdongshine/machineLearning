#coding=utf-8
import numpy as np
import matplotlib as mpl
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
## 设置字符集，防止中文乱码

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#创建数据

#产生X=（随机数在5以内的数据，6行100列二位数组）
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])

gnb = GaussianNB()
gnb.fit(X,y)
y_gnb = gnb.predict(X[2:3])
print "高斯贝叶斯结果：",y_gnb

mnb = MultinomialNB()
mnb.fit(X,y)
y_mnb = mnb.predict(X[2:3])
print "多项式贝叶斯结果：",y_mnb

bnb = BernoulliNB()
bnb.fit(X,y)
y_bnb = bnb.predict(X[2:3])
print "伯努利贝叶斯结果：",y_bnb