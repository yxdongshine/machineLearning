#coding=utf-8
from sklearn import metrics
from time import time

import numpy as np
import matplotlib as mpl
from prompt_toolkit import input
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm.classes import SVC, LinearSVC
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#加载数据
load_start_time = time()
remove = ('headers', 'footers', 'quotes')
#类别
categories = 'alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'
data_train = fetch_20newsgroups(data_home='../../data/',subset='train', categories=categories, shuffle=True, random_state=0, remove=remove)
data_test  = fetch_20newsgroups(data_home='../../data/',subset='test', categories=categories, shuffle=True, random_state=0, remove=remove)
print u"完成数据加载过程耗时:%.3fs" % (time() - load_start_time)


#查看测试数据和训练数据大小
def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6
categories = data_train.target_names
data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print u'数据类型：', type(data_train)
print("%d文本数量 - %0.3fMB (训练数据集)" % (len(data_train.data), data_train_size_mb))
print("%d文本数量 - %0.3fMB (测试数据集)" % (len(data_test.data), data_test_size_mb))
print u'训练集和测试集使用的%d个类别的名称：' % len(categories)
print(categories)


#得到训练集合和测试集合
x_train = data_train.data
y_train = data_train.target
x_test = data_test.data
y_test = data_test.target

#y_train 代表0123等四类类别代表
#print y_train
#输出具体格式内用查看
print u' -- 前5个文本 -- '
for i in range(5):
    print u'文本%d(属于类别 - %s)：' % (i+1, categories[y_train[i]])
    print x_train[i]
    print '\n\n'

print "===========开始分词================="
#把文字转化数字 矢量化
vectorizer = TfidfVectorizer(input='content',stop_words='english', max_df=0.5, sublinear_tf=True)
x_train = vectorizer.fit_transform(data_train.data)# x_train是稀疏的，scipy.sparse.csr.csr_matrix
x_test = vectorizer.transform(data_test.data)
print u'训练集样本个数：%d，特征个数：%d' % x_train.shape
print u'停止词:\n',
print(vectorizer.get_stop_words())


print "===========各种模型比较================="
#具体训练模型步凑 提炼成一个函数
def benchmark(clf, name):
    print u'分类器：', clf

    alpha_can = np.logspace(-2, 1, 10)
    model = GridSearchCV(clf, param_grid={'alpha': alpha_can}, cv=5)
    m = alpha_can.size

    if hasattr(clf, 'alpha'):
        model.set_params(param_grid={'alpha': alpha_can})
        m = alpha_can.size

    if hasattr(clf, 'n_neighbors'):
        neighbors_can = np.arange(1, 15)
        model.set_params(param_grid={'n_neighbors': neighbors_can})
        m = neighbors_can.size

    if hasattr(clf, 'C'):
        C_can = np.logspace(1, 3, 3)
        model.set_params(param_grid={'C': C_can})
        m = C_can.size

    if hasattr(clf, 'C') & hasattr(clf, 'gamma'):
        C_can = np.logspace(1, 3, 3)
        gamma_can = np.logspace(-3, 0, 3)
        model.set_params(param_grid={'C': C_can, 'gamma': gamma_can})
        m = C_can.size * gamma_can.size

    if hasattr(clf, 'max_depth'):
        max_depth_can = np.arange(4, 10)
        model.set_params(param_grid={'max_depth': max_depth_can})
        m = max_depth_can.size

    t_start = time()
    model.fit(x_train, y_train)
    t_end = time()
    t_train = (t_end - t_start) / (5 * m)
    print u'5折交叉验证的训练时间为：%.3f秒/(5*%d)=%.3f秒' % ((t_end - t_start), m, t_train)
    print u'最优超参数为：', model.best_params_

    t_start = time()
    y_hat = model.predict(x_test)
    t_end = time()
    t_test = t_end - t_start
    print u'测试时间：%.3f秒' % t_test

    train_acc = metrics.accuracy_score(y_train, model.predict(x_train))
    test_acc = metrics.accuracy_score(y_test, y_hat)
    print u'训练集准确率：%.2f%%' % (100 * train_acc)
    print u'测试集准确率：%.2f%%' % (100 * test_acc)

    return t_train, t_test, 1 - train_acc, 1 - test_acc, name


#开始传提参数
clfs = [
    [RidgeClassifier(), 'Ridge'],
    [KNeighborsClassifier(), 'KNN'],
    [MultinomialNB(), 'MultinomialNB'],
    [BernoulliNB(), 'BernoulliNB'],
    [RandomForestClassifier(n_estimators=200), 'RandomForest'],
    [SVC(), 'SVM'],
    [LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-4), 'LinearSVC-l1'],
    [LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-4), 'LinearSVC-l2']
]

#开始训练
result = []
for clf,name in clfs:
    a = benchmark(clf,name)
    result.append(a)
    print '\n'

result = np.array(result)



#最后对训练时间 测试时间  训练得分 测试得分
result = [[x[i] for x in result] for i in range(5)]
training_time, test_time, training_err, test_err, clf_names = result

training_time = np.array(training_time).astype(np.float)
test_time = np.array(test_time).astype(np.float)
training_err = np.array(training_err).astype(np.float)
test_err = np.array(test_err).astype(np.float)

x = np.arange(len(training_time))
plt.figure(figsize=(10, 7), facecolor='w')
ax = plt.axes()
b0 = ax.bar(x+0.1, training_err, width=0.2, color='#77E0A0')
b1 = ax.bar(x+0.3, test_err, width=0.2, color='#8800FF')
ax2 = ax.twinx()
b2 = ax2.bar(x+0.5, training_time, width=0.2, color='#FFA0A0')
b3 = ax2.bar(x+0.7, test_time, width=0.2, color='#FF8080')
plt.xticks(x+0.5, clf_names)
plt.legend([b0[0], b1[0], b2[0], b3[0]], (u'训练集错误率', u'测试集错误率', u'训练时间', u'测试时间'), loc='upper left', shadow=True)
plt.title(u'新闻组文本数据分类及不同分类器效果比较', fontsize=18)
plt.xlabel(u'分类器名称')
plt.grid(True)
plt.tight_layout(2)
plt.show()