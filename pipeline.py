from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from collections import Counter
import os
import json

objectIDs = []

with open("met_openaccess_euro_paintings_objectIDs.txt", 'r') as f:
  for line in f:
    objectIDs.append(line.strip())


data = []


for objectID in objectIDs:

  image_path = "./euro_paintings/" + objectID + ".jpg"
  metadata_path = "./euro_paintings/" + objectID + ".json"

  if os.path.exists(image_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    print(f"read image {image_path}")

    H, W, _ = img.shape
    img = img.copy()

    def is_solid_line(line):
        return np.all((line == line[0]).all(axis=-1))

    # Detect solid borders
    borders = {}
    for name, line in {
        "top": img[0],
        "bottom": img[-1],
        "left": img[:, 0],
        "right": img[:, -1]
    }.items():
        if is_solid_line(line):
            borders[name] = tuple(line[0])  # store as immutable RGB tuple

    if not borders:
        valid_mask = np.ones((H, W), dtype=bool)  # All pixels valid
    else:
        # Most common border color
        border_color = Counter(borders.values()).most_common(1)[0][0]
        border_color = np.array(border_color)

        # Create mask of pixels equal to border color
        border_mask = np.all(img == border_color, axis=-1)
        valid_mask = ~border_mask  # Keep non-border pixels

    # Run k-means only on valid pixels
    print(img.shape)
    image = img[valid_mask].reshape(-1, 3)
    print(image.shape)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    # Simple saliency mask: high sat + mid-to-high brightness
    mask = (saturation > 0.3) & (value > 0.2)
    salient_pixels = img[mask]

    if salient_pixels.shape[0] == 0:
      kmeans = KMeans(n_clusters=7, random_state=42).fit(image)
    else:
      kmeans = KMeans(n_clusters=7, random_state=42).fit(salient_pixels)
    colors = kmeans.cluster_centers_.astype(int)

    with open(metadata_path, "r") as f:
      metadata = json.load(f)
    begin_date = metadata["objectBeginDate"]
    end_date = metadata["objectEndDate"]
    date = (begin_date + end_date) // 2

    object_data = {}
 
    object_data["date"] = date
    object_data["title"] = metadata["title"]
    object_data["artist"] = metadata["artistDisplayName"]
    object_data["image_path"] = image_path
    object_data["colors"] = []

    for dominant_color in colors:
      rgb = {}
      rgb["r"] = int(dominant_color[0])
      rgb["g"] = int(dominant_color[1])
      rgb["b"] = int(dominant_color[2])
    
      object_data["colors"].append(rgb)
    
    data.append(object_data)

with open("data.json", "w") as f:
  json.dump(sorted(data, key=lambda x: x["date"]), f, indent=4, ensure_ascii=False)
