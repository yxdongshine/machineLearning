#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pylab as pl
import pandas as pd

from sklearn import svm
from sklearn import linear_model
from sklearn import tree

from sklearn.metrics import confusion_matrix

x_min, x_max = 0, 15
y_min, y_max = 0, 10
step = .1
# to plot the boundary, we're going to create a matrix of every possible point
# 把两个一维的映射成一个二维的矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
print 'xx, yy',xx, yy
#类似scala的那种，映射成表结构
df = pd.DataFrame(data={'x': xx.ravel(), 'y': yy.ravel()})

#画个圆
df['color_gauge'] = (df.x - 7.5) ** 2 + (df.y - 5) ** 2
#给圆上色
df['color'] = df.color_gauge.apply(lambda x: "red" if x<= 15 else "green")
#归一化
df['color_as_int'] = df.color.apply(lambda x: 0 if x == "red" else 1)

#从第1图开始
figure = 1

# plot a figure for the entire dataset
#df.color.unique()提取出不同的颜色
#开始画第一个图
for color in df.color.unique():
    idx = df.color == color
    pl.subplot(2, 2, figure)
    pl.scatter(df[idx].x, df[idx].y, color=color)
    pl.title('Actual')

train_idx = df.x<10
print 'df',df
train = df[train_idx]
test = df[-train_idx]

# train using the x and y position coordiantes
cols = ["x", "y"]

clfs = {
    "SVM": svm.SVC(degree=0.5),
    "Logistic": linear_model.LogisticRegression(),
    "Decision Tree": tree.DecisionTreeClassifier()
}

# racehorse different classifiers and plot the results
#clf_name对应名字，clf对应后面的
for clf_name, clf in clfs.iteritems():
    figure += 1
    print clf_name
    print 'figure',figure
    print 'clf',clf
    # train the classifier
    clf.fit(train[cols], train.color_as_int)

    # get the predicted values from the test set
    test['predicted_color_as_int'] = clf.predict(test[cols])
    test['pred_color']= test.predicted_color_as_int.apply(lambda x: "red" if x == 0 else "green")
    # create a new subplot on the plot
    pl.subplot(2, 2, figure)
    # plot eh predicted colorac
    for color in test.pred_color.unique():
        # plot only rows where pred_color is equal to color
        idx = test.pred_color == color
        pl.scatter(test[idx].x, test[idx].y, color=color)


        # plot the training set as well
    for color in train.color.unique():
        idx = train.color == color
        pl.scatter(train[idx].x, train[idx].y, color=color)

        # add a dotted line to show the boundary between the training and test set
    # (everything to the right of the line is in the test set)
    # this plots a vertical line
    train_line_y = np.linspace(y_min, y_max)  # evenly spaced array from 0 to 10
    train_line_x = np.repeat(10, len(train_line_y))
    # repeat 10 (threshold for traininset) n times
    # add a black, dotted line to the subplot
    pl.plot(train_line_x, train_line_y, 'k--', color="black")

    pl.title(clf_name)

    print "Confusion Matrix for %s:" % clf_name
    print confusion_matrix(test.color, test.pred_color)
    pl.show()