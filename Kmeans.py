import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# 13*13
# 26*26
# 52*52

dir = r"C:\Users\Jignesh\Desktop\Summer_Intern\IIIT-AR-13K_dataset\labels\train"

labels = os.listdir(dir)

X = []
Y = []
n = len(labels)
for i in range(0,n):
    label = labels[i]
    file = open(os.path.join(dir, label), 'r')
    lines = file.readlines()
    for line in lines:
        X.append(640*float(line.split()[-2]))
        Y.append(640*float(line.split()[-1]))

plt.scatter(X, Y, c = 'b')
plt.xlabel('Weidth')
plt.ylabel('Height')
plt.show(block=False)
plt.pause(5)
plt.close()

data = np.array([np.array(X), np.array(Y)])
data = data.T
scaler = StandardScaler()

data_t = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3).fit(data_t)
# print(kmeans.labels_)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers)

plt.scatter(data[:,0], data[:,1], c = labels, cmap = 'rainbow')
plt.scatter(centers[:,0], centers[:,1], c = 'black', s = 200, alpha = 0.5)
plt.xlabel('Weidth')
plt.ylabel('Height')
plt.show(block=False)
plt.pause(20)
plt.close()


data_split = [[],[],[]]
for i in range(0,len(labels)):
    data_split[labels[i]].append(data[i])

data_split = np.array(data_split)

data_split[0] = np.array(data_split[0])
data_split[1] = np.array(data_split[1])
data_split[2] = np.array(data_split[2])

data_split_t = [[],[],[]]
for i in range(0,3):
    data_split_t[i] = scaler.transform(data_split[i])

data_split_t = np.array(data_split_t)

labels_split = [[],[],[]]
centers_split = [[],[],[]]
for i in range(0,3):
    xyz = KMeans(n_clusters=3).fit(data_split_t[i])
    labels_split[i] = xyz.labels_
    centers_split[i] = xyz.cluster_centers_
    centers_split[i] = scaler.inverse_transform(centers_split[i])

print(centers_split)

# plot all the clusters in one figure
plt.scatter(data_split[0][:,0], data_split[0][:,1], c = labels_split[0], cmap = 'rainbow')
plt.scatter(centers_split[0][:,0], centers_split[0][:,1], c = 'black', s = 200, alpha = 0.5)
plt.scatter(data_split[1][:,0], data_split[1][:,1], c = labels_split[1], cmap = 'rainbow')
plt.scatter(centers_split[1][:,0], centers_split[1][:,1], c = 'black', s = 200, alpha = 0.5)
plt.scatter(data_split[2][:,0], data_split[2][:,1], c = labels_split[2], cmap = 'rainbow')
plt.scatter(centers_split[2][:,0], centers_split[2][:,1], c = 'black', s = 200, alpha = 0.5)
plt.xlabel('Weidth')
plt.ylabel('Height')
plt.show()
