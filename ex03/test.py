import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Chargement des données
data = np.load('data.npy')

inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sil_score = silhouette_score(data, kmeans.labels_)
    sil_scores.append(sil_score)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

calinski_scores = []
for i in range(2, 11):
    agglo = AgglomerativeClustering(n_clusters=i, affinity='manhattan', linkage='average')
    agglo.fit(data_scaled)
    calinski_scores.append(calinski_harabasz_score(data_scaled, agglo.labels_))

davies_scores = []
for i in range(2, 11):
    agglo = AgglomerativeClustering(n_clusters=i, affinity='manhattan', linkage='average')
    agglo.fit(data_scaled)
    davies_scores.append(davies_bouldin_score(data_scaled, agglo.labels_))


figure, axis = plt.subplots(2, 2, figsize=(10, 10))

# K-Means clustering avec distance euclidienne et méthode du coude
axis[0, 0].plot(range(1, 11), inertias)
axis[0, 0].set_title('Méthode du coude (K Means)')
axis[0, 0].set_xlabel('Nombre de clusters')
axis[0, 0].set_ylabel('Inertie')

# K Means  avec distance euclidienne et méthode du coeff de silhouette
axis[0, 1].plot(range(2, 11), sil_scores)
axis[0, 1].set_title('Méthode du coefficient de silhouette (K means)')
axis[0, 1].set_xlabel('Nombre de clusters')
axis[0, 1].set_ylabel('Score de silhouette')

# AgglomerativeClustering with Calinski-Harabasz index
axis[1, 0].plot(range(2, 11), calinski_scores)
axis[1, 0].set_title('Calinski-Harabasz (Agglomerative clustering)')
axis[1, 0].set_xlabel('Nombre de clusters')
axis[1, 0].set_ylabel('Score de Calinski-Harabasz')

# AgglomerativeClustering with Davies-Bouldin index
axis[1, 1].plot(range(2, 11), davies_scores)
axis[1, 1].set_title('INdice de Davies-Bouldin (Agglomerative clustering)')
axis[1, 1].set_xlabel('Nombre de clusters')
axis[1, 1].set_ylabel('Score de Davies-Bouldin')

plt.show()
