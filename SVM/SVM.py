from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X = X[Y!=2]
Y = Y[Y!=2]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

#svm_clf =SVC(kernel = "linear")
#svm_clf =SVC(kernel = "rbf")
svm_clf =SVC(kernel = "poly",degree=3)
svm_clf.fit(X_train,Y_train)

y_pre = svm_clf.predict(X_test)
print('iris验证集正确率：',(y_pre == Y_test).sum()/len(Y_test))

sonar = pd.read_csv('sonar.all-data', header=None, sep=',')
sonar1 = sonar.iloc[0:208, 0:60]
X2 = np.array(sonar1)
Y2 = np.zeros(208)
Y2[sonar.iloc[:, 60] == 'R'] = 1

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2,test_size=0.3)

svm_clf2 =SVC(kernel = "linear")
svm_clf2.fit(X2_train,Y2_train)

y2_pre = svm_clf2.predict(X2_test)
print('sonar验证集正确率：',(y2_pre == Y2_test).sum()/len(Y2_test))