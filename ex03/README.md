**Exercice 3.**

Pour cet exercice, nous avons choisi les méthodes de cluster KMeans et AgglomerativeClustering. Pour les metriques, nous avons utilisé la distance euclidienne pour le Kmeans, et le manhattan distance pour le AgglomerativeClustering. Pour les heuristiques, nous avons utilisé la méthode du coude (elbow method) et la méthode de la silhouette.

Voici les résultats de la méthode du coude pour les 2 clusters: 
https://i.imgur.com/ZOlXrGl.png
https://i.imgur.com/sUAxCBB.png

Nous voyons que le coude se situe aux alentours de 5 clusters. 

Desormais, nous calculons le score de silhouette des 2 clusters. Voici les résultats:
https://i.imgur.com/y6AVcPT.png

Cette fois ci, nous apercevons clairement que le coude se situe à 6, contrairement à la méthode du coude qui indique 5. Le problème avec la méthode du coude est que l'on aperçoit pas très clairement si l'on doit choisir la valeur 5 ou 6 sur la courbe. Avec la méthode de la silhouette, il n'y a aucun doute.

Pour vérifier ce résultat, nous affichons le résultat du KMeans avec 6 clusters: 

https://i.imgur.com/ZcDL2H5.png

On aperçoit bien 6 clusters distincts, ce qui confirme les résultats de la méthode de la silhouette.

En conclustion, la méthode de la silhouette est plus précise pour detecter le nombre de clusters nécéssaires. La méthode du coude elle, peut porter à confusion sur la valeur dont nous avons besoin. 

Code (disponible sur le repository avec le requirements.txt) 


```python
import numpy as np
from  sklearn.cluster  import  KMeans, AgglomerativeClustering
from  sklearn.preprocessing  import  MinMaxScaler
from  scipy.spatial.distance  import  cdist
from  sklearn.metrics  import  silhouette_score
import  matplotlib.pyplot  as  plt
from  yellowbrick.cluster  import  KElbowVisualizer

data = np.load('data.npy')

def  kmeans_euclidian_inertia():
	inertias = []
	kmeans = KMeans()
	results = KElbowVisualizer(kmeans, k=(1,12))
	results.fit(data)
	results.show()

kmeans_euclidian_inertia()

def  kmeans_euclidian_sil_scores():
	sil_scores = []
	for  k  in  range(2, 11):
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(data)
	sil_score = silhouette_score(data, kmeans.labels_)
	sil_scores.append(sil_score)
	return  sil_scores

sil_scores = kmeans_euclidian_sil_scores()

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def  elbow_agglo():
	agglo = AgglomerativeClustering(metric='manhattan', linkage='average')
	agglo.fit(data_scaled)
	results = KElbowVisualizer(agglo, k=(1,12))
	results.fit(data)
	results.show()

elbow_agglo()

silhou = []
for  i  in  range(2, 11):
	agglo = AgglomerativeClustering(n_clusters=i, metric='manhattan', linkage='average')
	agglo.fit(data_scaled)
	silhou.append(silhouette_score(data_scaled, agglo.labels_))


figure, axis = plt.subplots(2, 2, figsize=(10, 10))


# K Means avec distance euclidienne et méthode du coeff de silhouette
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

# verification nombre clusters
kmeans = KMeans(n_clusters=6)
kmeans.fit(data)
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.show()

```

