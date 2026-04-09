import pandas as pd
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# 1. Path Setup and Data Loading
# -----------------------------
DATA_PATH = r"D:\PycharmProjects\5015ECG\all_combined.csv"
SAVE_DIR = r"D:\PycharmProjects\5015ECG\unsupervised"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. Define Feature Groups (Two Modalities)
# -----------------------------
# Modality A: Cardiac electrical activity (HRV features)
ecg_features = [
    'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF', 'HRV_SampEn'
]

# Modality B: Peripheral hemodynamics / blood pressure-related features
# Note: DC (Deceleration Capacity) is often associated with autonomic regulation
bp_features = [
    'DC', 'BMI', 'HRV_GI', 'HRV_SI'
]

# Ensure columns exist and remove missing values
cols = ecg_features + bp_features + ['Age_group']
df_clean = df[cols].dropna()

X_ecg = df_clean[ecg_features]
Y_bp = df_clean[bp_features]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ecg)
Y_scaled = scaler.fit_transform(Y_bp)

# -----------------------------
# 3. Canonical Correlation Analysis (CCA)
# -----------------------------
# Number of canonical components (typically limited by smaller feature set)
n_components = 2
cca = CCA(n_components=n_components)

X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

# Compute canonical correlations
correlations = [
    np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
    for i in range(n_components)
]

# -----------------------------
# 4. Visualization: Cross-modal Relationship
# -----------------------------
plt.figure(figsize=(14, 6))

# Scatter plot for first canonical component
plt.subplot(1, 2, 1)
sns.regplot(
    x=X_c[:, 0],
    y=Y_c[:, 0],
    scatter_kws={'alpha': 0.3, 'color': '#3498DB'},
    line_kws={'color': 'red'}
)
plt.title(f"Canonical Correlation Mode 1 (r = {correlations[0]:.3f})")
plt.xlabel("ECG Component (Cardiac)")
plt.ylabel("BP Component (Peripheral)")

# Relationship with age groups
plt.subplot(1, 2, 2)
df_clean['CCA_Coord'] = X_c[:, 0]
sns.boxplot(
    x='Age_group',
    y='CCA_Coord',
    data=df_clean,
    palette='magma'
)
plt.title("Cardiac-Peripheral Coupling across Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Coupling Strength (CCA Component)")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cca_cross_modal.png"), dpi=300)
plt.show()

# -----------------------------
# 5. Feature Contribution Analysis (Loadings)
# -----------------------------
print("\n" + "=" * 50)
print("Cross-modal Relationship Report (CCA Analysis)")
print("=" * 50)

print(f"First canonical correlation: {correlations[0]:.4f}")

# ECG feature loadings
ecg_weights = pd.DataFrame(
    cca.x_loadings_,
    index=ecg_features,
    columns=['Mode 1', 'Mode 2']
)

print("\n--- ECG Feature Contributions (Mode 1) ---")
print(ecg_weights['Mode 1'].sort_values(ascending=False))

# BP feature loadings
bp_weights = pd.DataFrame(
    cca.y_loadings_,
    index=bp_features,
    columns=['Mode 1', 'Mode 2']
)

print("\n--- Peripheral Feature Contributions (Mode 1) ---")
print(bp_weights['Mode 1'].sort_values(ascending=False))