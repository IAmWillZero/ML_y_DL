import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

clientes = pd.DataFrame({"saldo":[50000, 45000, 48000, 43500, 47000, 52000, 
                                 20000, 26000, 25000, 23000, 21400, 18000,
                                   8000, 12000, 6000, 14500, 12600, 7000 ],
                        "transacciones": [25,20, 16, 23, 25, 18, 23, 22, 24, 21, 27, 18, 
                                          8, 3, 6, 4, 9, 3]})


escalador  =  MinMaxScaler().fit(clientes.values)

clientes = pd.DataFrame(escalador.transform(clientes.values), 
                        columns=["saldo", "transacciones"])

kmeans = KMeans(n_clusters=3).fit(clientes.values)
clientes["clusters"] = kmeans.labels_

#graficar

plt.figure(figsize=(6, 5), dpi=100)

colores = ["red", "blue", "orange", "black", "purple", "pink", "brown"]

for cluster in range(kmeans.n_clusters):
    plt.scatter(clientes[clientes["clusters"]== cluster]["saldo"], 
    clientes[clientes["clusters"]== cluster]["transacciones"],
                marker="o", s = 180, color= colores[cluster], alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[cluster][0],
                                        kmeans.cluster_centers_[cluster][1], 
                                        marker="P", s = 200, color = colores[cluster])
    
plt.title("Clientes", fontsize=20)
plt.xlabel("Saldo en cuenta de ahorros (soles)", fontsize = 15)
plt.ylabel("Transacciones realizadas en el mes", fontsize = 15)
plt.text(1.15, 0.2, "K = %i" % kmeans.n_clusters, fontsize=25)
plt.text(1.15, 0, "inercia = %0.2f" % kmeans.inertia_, fontsize=25)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()


