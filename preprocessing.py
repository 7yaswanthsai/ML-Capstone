import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
DATA_DIR = "C:/Users/91903/Documents/ML-Capstone/data"
VALID_EXTS = (".jpg", ".jpeg", ".png")

def load_and_preprocess_data():
    images, labels = [], []
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isalnum()])
    label_map = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_folder = os.path.join(DATA_DIR, cls)
        for img_name in os.listdir(cls_folder):
            if img_name.lower().endswith(VALID_EXTS):
                try:
                    img_path = os.join.path(cls_folder, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    labels.append(label_map[cls])
                except:
                    continue

    x = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = np.array(labels)

    return train_test_split(X, y, test_size=0.2, random_state=42), label_map