import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
from sklearn import datasets, metrics
import pandas as pd

class FCM(BaseEstimator, ClassifierMixin):
    def __init__(self, k, alpha=2):
        # method=2 => use L2 distance
        self.k = k
        self.alpha = alpha
        self.x = None
        self.y = None
        self.labels = None
        self.centers = None
        self.u = None
        self.iterations = 500

    def quick_L2(self, x, a):
        dis = -2 * np.dot(x, a.T)
        dis += np.einsum('ij,ij->i', x, x)[:, np.newaxis]
        dis += np.einsum('ij,ij->i', a, a)[np.newaxis, :]
        return dis

    def fit(self, x, y=None, init_method='u', seed=None, eps=1e-5):
        self.x = x
        self.y = y
        if seed is not None:
            np.random.seed(seed)
        if init_method == 'u':
            self.u = np.random.rand(self.x.shape[0], self.k)
            self.u /= np.sum(self.u, axis=1)[:, np.newaxis]
        else:
            pass

        pre_J = 0
        for i in range(self.iterations):
            u_a = self.u ** self.alpha  # u_{ij}^{\alpha}
            self.centers = np.dot(self.u.T, self.x) / np.sum(self.u, axis=0)[:, np.newaxis]

            dis = self.quick_L2(self.x, self.centers)
            J = np.sum(u_a * dis)
            if abs(J - pre_J) < eps:
                return
            e = 1 / (self.alpha - 1 + eps * 100)
            self.u = 1 / ((dis ** e) * np.sum(dis ** (-e), axis=1)[:, np.newaxis])
            pre_J = J

    def predict(self):
        return np.argmax(self.u, axis=1)


if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris['data']
    labels = iris['target']


    print('FCM:')
    fcm = FCM(k=3)
    fcm.fit(data)
    res = fcm.predict()
    print('iris-predict:')
    print(res)

    print('iris-SC指标: ' + str(metrics.silhouette_score(data, res, metric='euclidean')))

    sonar = pd.read_csv('sonar.all-data', header=None, sep=',')
    sonar1 = sonar.iloc[0:208, 0:60]
    data2 = np.array(sonar1)
    labels2 = np.zeros(208)
    labels2[sonar.iloc[:, 60] == 'R'] = 1

    fcm2 = FCM(k=2)
    fcm2.fit(data2)
    res2 = fcm2.predict()
    print('sonar-predict:')
    print(res2)

    print('sonar-SC指标: ' + str(metrics.silhouette_score(data2, res2, metric='euclidean')))
