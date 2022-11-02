from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sonar = pd.read_csv('sonar.all-data', header=None, sep=',')
sonar1 = sonar.iloc[0:208,0:60]
sonar2 = np.array(sonar1)
y1 = np.zeros(104)
y2 = np.ones(104)
y = np.append(y1, y2)

def KNN(k, X_tr, y_tr, X_te, y_te, distace_type):
        accuracy = 0
        for i in range(X_te.shape[0]):
                distance = np.zeros((2, X_tr.shape[0]))
                clas = np.zeros(2)
                for j in range(X_tr.shape[0]):
                        distance[1, j] = np.linalg.norm(X_te[i] - X_tr[j], distace_type)
                        distance[0, j] = y_tr[j]
                index = np.lexsort(distance)
                for l in range(k):
                        clas[int(distance[0,index[l]])] +=1
                prediction = clas.argmax()
                if prediction == y_te[i]:
                        accuracy +=1

        accuracy = accuracy/X_te.shape[0]
        return accuracy



accuracy_all = np.zeros((3,20))

def KNN_sonar(k, y, sonar2, distace_type):
        acc = 0
        for i in range(208):
                X_te = sonar2[i].reshape((1,60))
                X_tr = np.delete(sonar2, i, axis=0)
                y_te = y[i].reshape(1)
                y_tr = np.delete(y, i, axis=0)
                acc += KNN(k, X_tr, y_tr, X_te, y_te, distace_type)
        return acc/208


for m in tqdm(range(20)):
        k = m + 1
        accuracy_all[0][m] = KNN_sonar(k, y, sonar2, 1)
        accuracy_all[1][m] = KNN_sonar(k, y, sonar2, 2)
        accuracy_all[2][m] = KNN_sonar(k, y, sonar2, np.inf)
        pass

x = np.arange(1,21,1)
plt.title("KNN in sonar")
plt.xlabel('k')
plt.ylabel('accuracy')
plt.xlim((1, 20))
plt.ylim((0, 1.0))
plt.plot(x, accuracy_all[0], 'r-o', label = "distance_L1")
plt.plot(x, accuracy_all[1], 'g-o', label = "distance_L2")
plt.plot(x, accuracy_all[2], 'b-o', label = "distance_Lmax")
plt.legend()
plt.savefig('result of KNN in sonar.jpg')
plt.show()
