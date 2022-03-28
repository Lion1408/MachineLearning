from cProfile import label
from turtle import color
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, k_means
from yellowbrick.cluster import KElbowVisualizer

data = pd.read_csv("Mall.csv")

x = data.iloc[:, [3, 4]].values

plt.plot(x[:, 0], x[: , 1], 'o')
 
plt.show()

# Generate synthetic dataset with 8 random clusters
#X, y = make_blobs(n_samples=1000, n_features=12, centers=8, random_state=42)

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(2,11), metric='calinski_harabasz', timings=False, locate_elbow=True
)
visualizer.fit(x)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
k = list()
for i in range(1,11):
    k.append(KMeans(n_clusters = i).fit(x).inertia_)
plt.plot(range(1, 11), k, marker = 'o')
plt.show()
K = 5
y = KMeans(n_clusters = K).fit(x).fit_predict(x)
plt.plot(x[y==0][: , 0], x[y==0][: , 1],'bo', label = 'blue')
plt.plot(x[y==1][: , 0], x[y==1][: , 1],'co', label = 'cyan')
plt.plot(x[y==2][: , 0], x[y==2][: , 1],'yo', label = 'yellow')
plt.plot(x[y==3][: , 0], x[y==3][: , 1],'ro', label = 'red')
plt.plot(x[y==4][: , 0], x[y==4][: , 1],'go', label = 'green')
plt.plot(x[y==0][: , 0].mean(), x[y==0][: , 1].mean(),marker = 'X', color = 'black', label = 'center')
for i in range(1, K) :
    plt.plot(x[y==i][: , 0].mean(), x[y==i][: , 1].mean(),marker = 'X', color = 'black')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()