import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

def Fisher(X1, X2, n):
    X1 = X1[:, 0:n]
    X2 = X2[:, 0:n]
    m1 = (np.mean(X1, axis=0))
    m2 = (np.mean(X2, axis=0))
    m1 = m1.reshape(n, 1)  # 将行向量转换为列向量以便于计算
    m2 = m2.reshape(n, 1)

    # 计算类内离散度矩阵
    S1 = np.zeros((n, n))  # m1 = within_class_scatter_matrix1
    S2 = np.zeros((n, n))  # m2 = within_class_scatter_matrix2
    for i in range(0, X1.shape[0]):
        S1 += (X1[i].reshape(n, 1) - m1).dot((X1[i].reshape(n, 1) - m1).T)
    for i in range(0, X2.shape[0]):
        S2 += (X2[i].reshape(n, 1) - m2).dot((X2[i].reshape(n, 1) - m2).T)
    # 计算总类内离散度矩阵S_w
    S_w = S1 + S2

    # 计算最优投影方向 W
    W = np.linalg.inv(S_w).dot(m1 - m2)
    # 在投影后的一维空间求两类的均值
    m_1 = (W.T).dot(m1)
    m_2 = (W.T).dot(m2)

    # 计算分类阈值 W0(为一个列向量)
    W0 = 0.5 * (m_1 + m_2)

    return W, W0

def Classify(X, W, W0, n):
    y = (W.T).dot(X[0:n, :]) - W0
    return y

P1 = iris.data[0:50, 0:4]
P2 = iris.data[50:100, 0:4]
P3 = iris.data[100:150, 0:4]

result = np.zeros(100)
Accuracy = np.zeros(4)

for n in range(1, 5):
    count = 0
    for i in range(100):
        if i <= 49:
            test = P1[i]
            test = test.reshape(4, 1)
            train = np.delete(P1, i, axis=0)
            W, W0 = Fisher(train, P2, n)
            if (Classify(test, W, W0, n)) >= 0:
                count += 1
                result[i] = Classify(test, W, W0, n)
        else:
            test = P2[i-50]
            test = test.reshape(4, 1)
            train = np.delete(P2, i-50, axis=0)
            W, W0 = Fisher(P1, train, n)
            if (Classify(test, W, W0, n)) < 0:
                count += 1
                result[i] = Classify(test, W, W0, n)
    Accuracy[n-1] = count/100
    print("维数取%d时第一类和二类的分类准确率为:%.3f" % (n, Accuracy[n-1]))

#画曲线图
x = np.arange(1,5,1)
plt.xlabel('dimension')
plt.ylabel('Accuracy')
plt.xlim((1, 4))
plt.ylim((0, 1.0))
plt.plot(x, Accuracy, 'r-o')
plt.show()