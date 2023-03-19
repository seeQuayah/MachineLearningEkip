import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

# Chargement des données
data = np.load('data.npy')

def kmeans_euclidian_inertia():
    inertias = []
    kmeans = KMeans()
    results = KElbowVisualizer(kmeans, k=(1,12))
    results.fit(data)
    results.show()


kmeans_euclidian_inertia()

def kmeans_euclidian_sil_scores():
    sil_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        sil_score = silhouette_score(data, kmeans.labels_)
        sil_scores.append(sil_score)
    return sil_scores
sil_scores = kmeans_euclidian_sil_scores()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


def elbow_agglo():
    agglo = AgglomerativeClustering(metric='manhattan', linkage='average')
    agglo.fit(data_scaled)
    results = KElbowVisualizer(agglo, k=(1,12))
    results.fit(data)
    results.show()

elbow_agglo()

silhou = []
for i in range(2, 11):
    agglo = AgglomerativeClustering(n_clusters=i, metric='manhattan', linkage='average')
    agglo.fit(data_scaled)
    silhou.append(silhouette_score(data_scaled, agglo.labels_))


figure, axis = plt.subplots(2, 2, figsize=(10, 10))


# K Means  avec distance euclidienne et méthode du coeff de silhouette
axis[0, 0].plot(range(2, 11), sil_scores)
axis[0, 0].set_title('Méthode du coefficient de silhouette (K means)')
axis[0, 0].set_xlabel('Nombre de clusters')
axis[0, 0].set_ylabel('Score de silhouette')



# AgglomerativeClustering with silhouette score
axis[0, 1].plot(range(2, 11), silhou)
axis[0, 1].set_title('Méthode du coefficient de silhouette (Agglomerative clustering)')
axis[0, 1].set_xlabel('Nombre de clusters')
axis[0, 1].set_ylabel('Score de silhouette')

plt.show()


kmeans = KMeans(n_clusters=6)
kmeans.fit(data)
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.show()
