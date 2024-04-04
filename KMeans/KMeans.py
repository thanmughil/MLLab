import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("../csv/URL_data.csv")
X = df.drop(columns=['target'])
kmeans = KMeans(n_clusters=2, random_state=42) 
kmeans.fit(X)
cluster_labels = kmeans.labels_

print("Silhouette Score:", round(silhouette_score(X, cluster_labels),2))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()