import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Cargar el conjunto de datos Iris
iris = load_iris()
data = iris.data
target = iris.target

# 2. Crear un DataFrame
df = pd.DataFrame(data=data, columns=iris.feature_names)

# 3. Normalizar los datos (solo longitud y ancho del pétalo)
scaler = MinMaxScaler().fit(df[['petal length (cm)', 'petal width (cm)']])
df_normalized = pd.DataFrame(scaler.transform(df[['petal length (cm)', 'petal width (cm)']]), 
                              columns=['petal length (cm)', 'petal width (cm)'])

# 4. Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42).fit(df_normalized.values)
df['Cluster'] = kmeans.labels_

# 5. Calcular el Silhouette Score
silhouette_avg = silhouette_score(df_normalized.values, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# 6. Visualizar resultados
plt.figure(figsize=(8, 6), dpi=100)

# Graficar los clústeres
colores = ["red", "blue", "green"]
for cluster in range(kmeans.n_clusters):
    plt.scatter(df[df["Cluster"] == cluster]["petal length (cm)"], 
                df[df["Cluster"] == cluster]["petal width (cm)"],
                marker="o", s=100, color=colores[cluster], alpha=0.5, label=f'Cluster {cluster + 1}')

# Graficar los centroides
centroids = kmeans.cluster_centers_
centroids_inverse = scaler.inverse_transform(centroids)
plt.scatter(centroids_inverse[:, 0], centroids_inverse[:, 1], 
            marker="X", s=300, color="black", label="Centroides")

plt.title("Agrupamiento de Flores Iris con K-means (Longitud y Ancho del Pétalo)", fontsize=20)
plt.xlabel("Longitud del Pétalo (cm)", fontsize=15)
plt.ylabel("Ancho del Pétalo (cm)", fontsize=15)
plt.legend()
plt.grid()
plt.show()
