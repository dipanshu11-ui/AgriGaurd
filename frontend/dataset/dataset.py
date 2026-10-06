import os
import pickle
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# 1. Dataset ka path auto-detect
if os.path.exists("Dataset/PlantVillage"):
    dataset_path = "Dataset/PlantVillage"
elif os.path.exists("Dataset"):
    dataset_path = "Dataset"
else:
    dataset_path = "../Dataset/PlantVillage"

print(f"Using dataset: {dataset_path}")

X = []
y = []
LIMIT = 400  # har class se 400 photo - fast + accurate

for cls_name in os.listdir(dataset_path):
    cls_path = os.path.join(dataset_path, cls_name)
    if not os.path.isdir(cls_path):
        continue
    
    count = 0
    for img_name in os.listdir(cls_path):
        if count >= LIMIT:
            break
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(cls_path, img_name)
                img = Image.open(img_path).convert('RGB').resize((64, 64))
                arr = np.array(img)
                # Color + Texture feature
                feat = [
                    np.mean(arr[:,:,0]), 
                    np.mean(arr[:,:,1]), 
                    np.mean(arr[:,:,2]),
                    np.std(arr)
                ]
                X.append(feat)
                y.append(cls_name)
                count += 1
            except:
                continue
    print(f"{cls_name}: {count} images loaded")

print(f"\nTotal Images: {len(X)}")

# 2. Model Training
print("Training start...")
model = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
model.fit(X, y)

# 3. Save model
os.makedirs("backend", exist_ok=True)
save_path = "backend/model.pkl"
if not os.path.exists("backend"):
    save_path = "model.pkl"

with open(save_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ HO GAYA! Model save ho gaya -> {save_path}")
print(f"Size: {os.path.getsize(save_path)/1024/1024:.2f} MB")