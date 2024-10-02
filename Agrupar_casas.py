import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Crear un DataFrame simulado con características de casas
data = {
    "tamaño_m2": [100, 120, 80, 150, 90, 200, 250, 130, 110, 95],
    "habitaciones": [3, 3, 2, 4, 2, 5, 5, 3, 3, 2],
    "baños": [2, 2, 1, 3, 1, 4, 4, 2, 2, 1],
    "precio": [300000, 350000, 200000, 500000, 250000, 600000, 700000, 400000, 330000, 220000]
}
casas = pd.DataFrame(data)

# 2. Normalizar los datos
scaler = MinMaxScaler().fit(casas.values)
casas_normalized = pd.DataFrame(scaler.transform(casas.values), columns=casas.columns)

# 3. Aplicar K-means
k = 3  # Número de clústeres
kmeans = KMeans(n_clusters=k, random_state=42).fit(casas_normalized.values)
casas['Cluster'] = kmeans.labels_

# 4. Calcular el Silhouette Score
silhouette_avg = silhouette_score(casas_normalized.values, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# 5. Visualizar resultados (usando tamaño y precio para la visualización)
plt.figure(figsize=(10, 6), dpi=100)

# Graficar los clústeres
colores = ["red", "blue", "green"]
for cluster in range(k):
    plt.scatter(casas[casas["Cluster"] == cluster]["tamaño_m2"], 
                casas[casas["Cluster"] == cluster]["precio"],
                marker="o", s=100, color=colores[cluster], alpha=0.5, label=f'Cluster {cluster + 1}')

# Graficar los centroides
centroids = kmeans.cluster_centers_
centroids_inverse = scaler.inverse_transform(centroids)
plt.scatter(centroids_inverse[:, 0], centroids_inverse[:, 3], 
            marker="X", s=300, color="black", label="Centroides")

plt.title("Agrupamiento de Casas con K-means", fontsize=20)
plt.xlabel("Tamaño (m²)", fontsize=15)
plt.ylabel("Precio (Soles)", fontsize=15)
plt.legend()
plt.grid()
plt.show()
