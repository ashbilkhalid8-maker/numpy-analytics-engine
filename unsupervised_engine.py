import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Generate Synthetic Customer Dataset
# Imagine 500 customers described by 5 different behavioral features (e.g., spending score, income, visit frequency, etc.)
print("--- 1. Generating Data ---")
X, _ = make_blobs(n_samples=500, n_features=5, centers=4, random_state=42)

# Standardize features (mean=0, variance=1) - crucial preprocessing step for PCA and K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Original Data Shape:", X_scaled.shape)

# 2. Apply PCA (Principal Component Analysis) for Dimensionality Reduction
# We reduce our 5-dimensional data down to 2 dimensions so we can analyze and visualize it easily
print("\n--- 2. Applying PCA ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Reduced PCA Data Shape:", X_pca.shape)
print("Explained Variance Ratio (Information retained):", np.sum(pca.explained_variance_ratio_) * 100, "%")

# 3. Apply K-Means Clustering
# We group the customers into 4 distinct clusters based on their patterns
print("\n--- 3. Applying K-Means Clustering ---")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Find the center points (centroids) of each cluster
centroids = kmeans.cluster_centers_
print("Cluster Centers Shape:", centroids.shape)
print("First few assigned cluster labels:", cluster_labels[:10])

# Summary of cluster distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} customers assigned")