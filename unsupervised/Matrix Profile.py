import pandas as pd
import numpy as np
import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import stumpy
import os

# -----------------------------
# Environment Configuration
# -----------------------------
SAVE_DIR = r"D:\PycharmProjects\5015ECG\unsupervised"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# -----------------------------
# 1. Generate Simulated RR Interval Series
# (including repeated patterns and anomalies)
# -----------------------------
def generate_rr_series():
    np.random.seed(42)

    # Base heartbeat (~800 ms with small fluctuations)
    n = 1000
    rr_series = 800 + np.random.normal(0, 10, n)

    # Insert Motif (repeated slowing pattern)
    motif_pattern = np.array([850, 900, 1000, 1100, 1000, 900, 850])

    # Insert motif at positions 200 and 600
    rr_series[200:207] = motif_pattern
    rr_series[600:607] = motif_pattern

    # Insert Discord (rare abnormal pattern)
    rr_series[850:857] = [800, 500, 450, 500, 800, 810, 800]

    return rr_series.astype(np.float64)


rr_data = generate_rr_series()

# Subsequence window length (about 30 heartbeats)
m = 30

# -----------------------------
# 2. Compute Matrix Profile
# -----------------------------
print("Computing Matrix Profile (pattern discovery)...")

# mp contains distance values (column 0 = minimum Euclidean distance)
mp = stumpy.stump(rr_data, m=m)

# Find Motif (smallest distance)
motif_idx = np.argsort(mp[:, 0])[0]
nearest_neighbor_idx = mp[motif_idx, 1]

# Find Discord (largest distance)
discord_idx = np.argsort(mp[:, 0])[-1]

# -----------------------------
# 3. Visualization
# -----------------------------
fig, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Plot 1: Raw RR interval series
ax[0].plot(rr_data, color='#2E86C1', label='RR Interval Series')
ax[0].set_title('A. Raw RR Interval Series', fontsize=14, fontweight='bold')

# Highlight Motif regions
ax[0].axvspan(motif_idx, motif_idx + m, color='green', alpha=0.3, label='Motif (Pattern)')
ax[0].axvspan(nearest_neighbor_idx, nearest_neighbor_idx + m, color='green', alpha=0.3)

# Highlight Discord region
ax[0].axvspan(discord_idx, discord_idx + m, color='red', alpha=0.3, label='Discord (Anomaly)')
ax[0].legend()

# Plot 2: Matrix Profile
ax[1].plot(mp[:, 0], color='#8E44AD')
ax[1].set_title('B. Matrix Profile (Lower distance = Higher similarity)', fontsize=14)
ax[1].axvline(motif_idx, color='green', linestyle='--')
ax[1].axvline(discord_idx, color='red', linestyle='--')

# Plot 3: Motif comparison
ax[2].plot(rr_data[motif_idx: motif_idx + m],
           color='green', label='Motif Instance 1')
ax[2].plot(rr_data[nearest_neighbor_idx: nearest_neighbor_idx + m],
           color='orange', linestyle='--',
           label='Motif Instance 2 (Matched)')
ax[2].set_title('C. Comparison of Discovered Motif Instances', fontsize=14)
ax[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "motif_discovery_results.png"), dpi=300)
print(f"Results saved to: {SAVE_DIR}")
plt.show()

# -----------------------------
# 4. Result Interpretation
# -----------------------------
print("\n" + "=" * 50)
print("Pattern Discovery Report")
print("=" * 50)

print(f"Most significant motif found at positions: {motif_idx} and {nearest_neighbor_idx}")
print(f"Most unusual pattern (discord) found at position: {discord_idx}")

print("\nAnalysis Insights:")
print("1. The Matrix Profile successfully identifies repeated 'deceleration-acceleration' patterns.")
print("2. This approach can automatically detect physiological rhythms such as Respiratory Sinus Arrhythmia (RSA).")