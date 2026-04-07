import os
import cv2
import numpy as np

INPUT_DIR = "data/known/Aditya"
OUTPUT_DIR = "data/known/Aditya"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment(img):
    augmented = []

    # Flip
    augmented.append(cv2.flip(img, 1))

    # Brightness
    for val in [30, -30]:
        bright = cv2.convertScaleAbs(img, alpha=1, beta=val)
        augmented.append(bright)

    # Rotation
    h, w = img.shape[:2]
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)

    # Blur
    augmented.append(cv2.GaussianBlur(img, (5,5), 0))

    return augmented

count = 0

for file in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file)

    img = cv2.imread(path)
    if img is None:
        continue

    aug_images = augment(img)

    for aug in aug_images:
        save_path = os.path.join(OUTPUT_DIR, f"aug_{count}.jpg")
        cv2.imwrite(save_path, aug)
        count += 1

print(f" Generated {count} augmented images")