#coding=utf-8
import pandas as pd
from sklearn import svm
from sklearn import metrics
input_file = '../data/moment.csv'
outputfile1 = '../data/cm_train.xls' #训练样本混淆矩阵保存路径
outputfile2 = '../data/cm_test.xls' #测试样本混淆矩阵保存路径
data = pd.read_csv(input_file)

#转化成矩阵
data = data.as_matrix()
# 前面的百分之八十作为训练集合 剩下的作为测试机集合


boundary = int(0.8*len(data))
train_data = data[:boundary, :]
#print  train_data.head()
test_data = data[boundary:, :]

# 构造特征x 和标签 y
x_train = train_data[:,2:]
y_train = train_data[:,0]

x_test = test_data[:,2:]
y_test = test_data[:,0]
#print  x_train[0]
#print  y_train[0]

# 开始训练模型
model = svm.SVC()
model.fit(x_train, y_train)

#生成混淆矩阵
#训练集中的混淆矩阵
cm_train = metrics.confusion_matrix(y_train,model.predict(x_train))
#测试集中的混淆矩阵
cm_test = metrics.confusion_matrix(y_test,model.predict(x_test))

print  cm_train

print ("--------------")

print  cm_test

#保存结果
pd.DataFrame(cm_train, index = range(0, 3), columns = range(0, 3)).to_excel(outputfile1)
pd.DataFrame(cm_test, index = range(0, 3), columns = range(0, 3)).to_excel(outputfile2)