import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.load('data.npy')
labels = np.load('labels.npy')

pca_2d = PCA(n_components=2)
reduced_data_2d = pca_2d.fit_transform(data)

x_data = reduced_data_2d[:, 0] - np.mean(reduced_data_2d[:, 0])
y_data = reduced_data_2d[:, 1] - np.mean(reduced_data_2d[:, 1])
plt.scatter(x_data, y_data, c=labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2D')
plt.show()


pca_3d = PCA(n_components=3)
reduced_data_3d = pca_3d.fit_transform(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_data = reduced_data_3d[:, 0] - np.mean(reduced_data_3d[:, 0])
y_data = reduced_data_3d[:, 1] - np.mean(reduced_data_3d[:, 1])
z_data = reduced_data_3d[:, 2] - np.mean(reduced_data_3d[:, 2])
ax.scatter(x_data, y_data, z_data, c=labels)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3D')
plt.show()