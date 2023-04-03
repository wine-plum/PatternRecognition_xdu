import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import datasets, metrics
import time
import pandas as pd

class KMeans(BaseEstimator, ClassifierMixin):
    def __init__(self, k):
        self.k = k
        self.x = None
        self.y = None
        self.labels = None
        self.centers = None
        self.iterations = 500

    def quick_L2(self, x, a):

        dis = -2 * np.dot(x, a.T)
        dis += np.einsum('ij,ij->i', x, x)[:, np.newaxis]
        dis += np.einsum('ij,ij->i', a, a)[np.newaxis, :]
        return dis

    def fit(self, x, y=None, init_method='random_point', seed=None, eps=1e-5):
        self.x = x
        self.y = y
        self.centers = 0

        if seed is not None:
            np.random.seed(seed)
        if init_method == 'random_point':
            self.centers = x[np.random.choice(x.shape[0], self.k), :]
        else:
            self.centers = np.random.randint(np.min(x), np.max(x), (x.shape[0], self.k))

        pre_centers = self.centers.copy()
        for i in range(self.iterations):
            dis = self.quick_L2(self.x, self.centers)
            idx = np.argmin(dis, axis=1)
            for j in range(self.centers.shape[0]):
                self.centers[j, :] = np.mean(self.x[idx == j, :], axis=0)
            if np.mean(np.abs(pre_centers - self.centers)) < eps:
                break
            pre_centers = self.centers.copy()

    def predict(self, a=None):
        if a is None:
            a = self.x
        dis = self.quick_L2(a, self.centers)
        idx = np.argmin(dis, axis=1)
        return idx

if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris['data']
    labels = iris['target']

    t = time.perf_counter()
    kmeans = KMeans(k=3)
    kmeans.fit(data)
    res = kmeans.predict()
    print('iris-predict:')
    print(res)

    print('iris-SC指标: ' + str(metrics.silhouette_score(data, res, metric='euclidean')))



    sonar = pd.read_csv('sonar.all-data', header=None, sep=',')
    sonar1 = sonar.iloc[0:208, 0:60]
    data2 = np.array(sonar1)
    labels2 = np.zeros(208)
    labels2[sonar.iloc[:, 60] == 'R'] = 1

    kmeans2 = KMeans(k=2)
    kmeans2.fit(data2)
    res2 = kmeans2.predict()
    print('sonar-predict:')
    print(res2)

    print('sonar-SC指标: ' + str(metrics.silhouette_score(data2, res2, metric='euclidean')))

