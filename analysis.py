import json
import colorsys
import numpy as np

# Load your JSON data
with open("data.json", "r") as f:
    paintings = json.load(f)

# Helper: convert RGB to hue [0, 1]
def rgb_to_hue(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h  # hue ∈ [0, 1]

# Helper: compute circular mean and variance of hues
def circular_hue_stats(hues):
    hues_rad = np.array(hues) * 2 * np.pi
    sin_sum = np.sum(np.sin(hues_rad))
    cos_sum = np.sum(np.cos(hues_rad))
    mean_angle = np.arctan2(sin_sum, cos_sum)
    mean_hue = (mean_angle % (2 * np.pi)) / (2 * np.pi)

    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(hues_rad)
    variance = 1 - R
    return mean_hue, variance

# Process each painting
for painting in paintings:
    rgb_list = [(c["r"], c["g"], c["b"]) for c in painting["colors"]]
    hues = [rgb_to_hue(r, g, b) for r, g, b in rgb_list]
    mean_hue, hue_variance = circular_hue_stats(hues)
    
    painting["mean_hue"] = mean_hue
    painting["hue_variance"] = hue_variance

# Example output
for p in paintings[:3]:  # first 3 paintings
    print(f"{p['title']} (by {p['artist']})")
    print(f"  Mean Hue: {round(p['mean_hue'], 3)}")
    print(f"  Hue Variance: {round(p['hue_variance'], 3)}\n")

import matplotlib.pyplot as plt

# Extract dates, mean hues, and variances
dates = [p["date"] for p in paintings if "mean_hue" in p]
mean_hues = [p["mean_hue"] for p in paintings if "mean_hue" in p]
hue_variances = [p["hue_variance"] for p in paintings if "hue_variance" in p]

# Plot mean hue over time
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(dates, mean_hues, alpha=0.7, c=mean_hues, cmap='hsv')
plt.title("Mean Hue Over Time")
plt.xlabel("Year")
plt.ylabel("Mean Hue (0–1)")
plt.colorbar(label="Hue")
plt.grid(True)

# Plot hue variance over time
plt.subplot(1, 2, 2)
plt.scatter(dates, hue_variances, alpha=0.7, color='orange')
plt.title("Hue Variance Over Time")
plt.xlabel("Year")
plt.ylabel("Hue Variance (0–1)")
plt.grid(True)

plt.tight_layout()
plt.show()

