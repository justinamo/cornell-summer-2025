import json
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from sklearn.linear_model import LinearRegression

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

# --- Group by nationality ---
from collections import defaultdict

by_nationality = defaultdict(list)

for p in paintings:
    if "mean_hue" in p and "nationality" in p:
        by_nationality[p["nationality"]].append(p)

# --- Compute stats per nationality ---
nationality_stats = {}

for nation, entries in by_nationality.items():
    if len(entries) < 5 or not nation:
        continue

    all_hues = []
    for p in entries:
        rgb_list = [(c["r"], c["g"], c["b"]) for c in p["colors"]]
        hues = [rgb_to_hue(r, g, b) for r, g, b in rgb_list]
        all_hues.extend(hues)

    if all_hues:
        mean_hue, variance = circular_hue_stats(all_hues)
        nationality_stats[nation] = {
            "mean_hue": mean_hue,
            "hue_variance": variance,
            "count": len(entries)
        }


# Sort by count or hue
sorted_nations = sorted(nationality_stats.items(), key=lambda x: x[1]["mean_hue"])

labels = [n for n, _ in sorted_nations]
mean_hues = [stats["mean_hue"] for _, stats in sorted_nations]
variances = [stats["hue_variance"] for _, stats in sorted_nations]

plt.figure(figsize=(10, 5))
plt.barh(labels, mean_hues, color='gray')
plt.xlabel("Mean Hue")
plt.title("Mean Hue by Nationality")
plt.tight_layout()
plt.show()


# Extract date and mean lightness from each painting
dates = []
mean_lightnesses = []

for painting in paintings:
    if "colors" not in painting or not painting["colors"]:
        continue

    # Convert RGB colors to LAB
    rgb_colors = np.array([[(
        color["r"] / 255.0,
        color["g"] / 255.0,
        color["b"] / 255.0
    ) for color in painting["colors"]]])

    lab_colors = rgb2lab(rgb_colors)[0]  # shape (N, 3)

    # Take the average L (lightness) value
    lightness_mean = np.mean(lab_colors[:, 0])  # L ∈ [0, 100]
    mean_lightnesses.append(lightness_mean)
    dates.append(painting["date"])

# Sort by date
sorted_data = sorted(zip(dates, mean_lightnesses))
dates_sorted, lightness_sorted = zip(*sorted_data)

# Convert to NumPy arrays
X = np.array(dates_sorted).reshape(-1, 1)
y = np.array(lightness_sorted)

# Fit linear regression
reg = LinearRegression().fit(X, y)
trend = reg.predict(X)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c=y, cmap="gray", alpha=0.6, label="Mean Lightness")
plt.plot(X, trend, color="red", linewidth=2, label="Trend Line")
plt.title("Mean Lightness Over Time")
plt.xlabel("Year")
plt.ylabel("Lightness (0–100)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def rgb_to_lightness(r, g, b):
    r_, g_, b_ = r / 255, g / 255, b / 255
    return (max(r_, g_, b_) + min(r_, g_, b_)) / 2  # lightness ∈ [0, 1]

# --- Group lightness values by nationality ---
painting_counts = defaultdict(int)
lightness_by_nation = defaultdict(list)

for p in paintings:
    nationality = p.get("nationality", "").strip()
    if not nationality or "colors" not in p:
        continue  # Skip if nationality is missing/blank or colors are missing

    painting_counts[nationality] += 1

    lightnesses = [
        rgb_to_lightness(c["r"], c["g"], c["b"])
        for c in p["colors"]
    ]
    lightness_by_nation[nationality].extend(lightnesses)

# --- Filter out nationalities with < 5 paintings ---
mean_lightness = {
    nation: np.mean(lightness_by_nation[nation])
    for nation in lightness_by_nation
    if painting_counts[nation] >= 5
}

# --- Sort and plot ---
sorted_items = sorted(mean_lightness.items(), key=lambda x: x[1])
labels = [k for k, _ in sorted_items]
means = [v for _, v in sorted_items]

plt.figure(figsize=(10, 6))
plt.barh(labels, means, color='gray')
plt.xlabel("Mean Lightness (0–1)")
plt.title("Mean Lightness by Nationality (≥ 5 paintings)")
plt.tight_layout()
plt.show()

def classify_hue(hue):
    if hue < 0.04 or hue > 0.95:
        return "red"
    elif 0.55 < hue < 0.72:
        return "blue"
    else:
        return "other"

# --- Aggregate red/blue proportions per painting ---
time_buckets = defaultdict(list)

for p in paintings:
    if "date" not in p or "colors" not in p:
        continue

    hues = [rgb_to_hue(c["r"], c["g"], c["b"]) for c in p["colors"]]
    labels = [classify_hue(h) for h in hues]

    total = len(labels)
    if total == 0:
        continue

    red_prop = labels.count("red") / total
    blue_prop = labels.count("blue") / total

    year = p["date"]
    decade = (year // 10) * 10  # or century = (year // 100) * 100

    time_buckets[decade].append((red_prop, blue_prop))

# --- Average by time bucket ---
sorted_years = sorted(time_buckets.keys())
red_means = [np.mean([x[0] for x in time_buckets[y]]) for y in sorted_years]
blue_means = [np.mean([x[1] for x in time_buckets[y]]) for y in sorted_years]

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(sorted_years, red_means, color="red", label="Red Usage")
plt.plot(sorted_years, blue_means, color="blue", label="Blue Usage")
plt.xlabel("Year")
plt.ylabel("Average Proportion")
plt.title("Red and Blue Color Usage Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

