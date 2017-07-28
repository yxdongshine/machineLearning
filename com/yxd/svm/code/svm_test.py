#coding=utf-8
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  svm
import matplotlib.colors
import matplotlib as mpl

from sympy.core.tests.test_arit import same_and_same_prec

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False



arrays = [np.random.randn(3, 4) for _ in range(10)]
print arrays.__len__()


xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

print xx.shape
print ("----------------------")
print yy.shape

# np.r_按row来组合array，
# np.c_按colunm来组合array

a = np.array([1,2,3])
b = np.array([5,2,5])
#测试 np.r_
np.r_[a,b]
#测试 np.c_
np.c_[a,b]
np.c_[a,[0,0,0],b]


[x,y,z]=peaks(25);
subplot(2,2,1);
contour(z);
title('contour函数效果');
subplot(2,2,2);
contourf(x,y,z);
title('contourf函数效果');
subplot(2,2,3);
pcolor(x,y,z);
title('pcolor函数效果');
subplot(2,2,4);
contour3(x,y,z);
title('contour3函数效果');