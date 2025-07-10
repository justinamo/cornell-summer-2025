from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
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
    image = Image.open(image_path).convert("RGB")
    print(f"read image {image_path}")

    image = image.resize((256, 256))

    # Convert to numpy and flatten
    pixels = np.array(image).reshape(-1, 3)
    
    # Apply k-means
    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans.fit(pixels)
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
  json.dump(data, f, indent=4, ensure_ascii=False)
