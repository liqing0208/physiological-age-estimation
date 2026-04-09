import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent GUI backend issues
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import umap
import os

# -----------------------------
# 1. Path Configuration
# -----------------------------
BASE_DIR = r"D:\PycharmProjects\5015ECG"
DATA_PATH = os.path.join(BASE_DIR, "all_combined.csv")
SAVE_DIR = os.path.join(BASE_DIR, "unsupervised")
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 2. Data Loading & Preprocessing
# -----------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Columns to exclude from features
exclude_cols = ['ID', 'Age_group', 'Sex', 'Device', 'age_range', 'Length']
features = [c for c in df.columns if c not in exclude_cols]

# Fill missing values with median
X = df[features].fillna(df[features].median())

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Dimensionality Reduction (UMAP + PCA)
# -----------------------------
print("Running UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

print("Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# 4. Clustering Methods
# -----------------------------
print("Clustering...")

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=2.5, min_samples=10)
db_labels = dbscan.fit_predict(X_scaled)

# -----------------------------
# 5. Evaluation (Silhouette Score)
# -----------------------------
def safe_silhouette(X, labels, name):
    unique_labels = set(labels)
    if len(unique_labels) > 1 and -1 not in unique_labels:
        score = silhouette_score(X, labels)
        print(f"{name} Silhouette Score: {score:.3f}")
    else:
        print(f"{name} cannot compute Silhouette (weak cluster structure)")

print("\nClustering Evaluation:")
safe_silhouette(X_scaled, kmeans_labels, "KMeans")
safe_silhouette(X_scaled, agg_labels, "Agglomerative")
safe_silhouette(X_scaled, db_labels, "DBSCAN")

# -----------------------------
# 6. Save Cluster Results
# -----------------------------
df['KMeans'] = kmeans_labels
df['Agglomerative'] = agg_labels
df['DBSCAN'] = db_labels
df['UMAP1'] = X_umap[:, 0]
df['UMAP2'] = X_umap[:, 1]

# -----------------------------
# 7. Visualization (Comparison Plot)
# -----------------------------
print("Generating comparison plots...")

plt.figure(figsize=(18, 5))

# KMeans
plt.subplot(1, 3, 1)
sns.scatterplot(x='UMAP1', y='UMAP2', hue='KMeans',
                palette='viridis', data=df, s=10)
plt.title("KMeans (k=3)")

# Agglomerative
plt.subplot(1, 3, 2)
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Agglomerative',
                palette='coolwarm', data=df, s=10)
plt.title("Agglomerative")

# DBSCAN
plt.subplot(1, 3, 3)
sns.scatterplot(x='UMAP1', y='UMAP2', hue='DBSCAN',
                palette='tab10', data=df, s=10)
plt.title("DBSCAN")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "clustering_comparison.png"), dpi=300)
plt.close()

# -----------------------------
# 8. Age Distribution Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Age_group',
                palette='magma', data=df, s=10)
plt.title("Age Distribution on UMAP")
plt.savefig(os.path.join(SAVE_DIR, "age_distribution.png"), dpi=300)
plt.close()

# -----------------------------
# 9. Cluster Analysis (Physiological Age Insight)
# -----------------------------
print("\nPhysiological Analysis:")

cluster_profile = df.groupby('KMeans')['Age_group'].mean().sort_values()

young_cid = cluster_profile.index[0]
old_cid = cluster_profile.index[-1]

print(f"Young physiological cluster ID: {young_cid}")
print(f"Older physiological cluster ID: {old_cid}")

rejuvenated = df[(df['Age_group'] >= 7) & (df['KMeans'] == young_cid)]
print(f"Number of rejuvenated individuals: {len(rejuvenated)}")

# -----------------------------
# 10. Save Final Results
# -----------------------------
output_path = os.path.join(SAVE_DIR, "all_with_clusters.csv")
df.to_csv(output_path, index=False)

print(f"\nAll results saved to: {SAVE_DIR}")
print("Unsupervised pipeline completed.")