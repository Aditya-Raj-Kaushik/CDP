import os
import cv2
import xml.etree.ElementTree as ET

# Paths
IMAGE_DIR = "data/mask_dataset/images"
ANNOTATION_DIR = "data/mask_dataset/annotations"

OUTPUT_DIR = "data/mask_dataset/images_classified"

WITH_MASK_DIR = os.path.join(OUTPUT_DIR, "with_mask")
WITHOUT_MASK_DIR = os.path.join(OUTPUT_DIR, "without_mask")

os.makedirs(WITH_MASK_DIR, exist_ok=True)
os.makedirs(WITHOUT_MASK_DIR, exist_ok=True)

count_mask = 0
count_no_mask = 0

for file in os.listdir(ANNOTATION_DIR):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(ANNOTATION_DIR, file))
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)

    for obj in root.findall("object"):
        label = obj.find("name").text

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        face = image[ymin:ymax, xmin:xmax]

        if face.size == 0:
            continue

        if "mask" in label.lower():
            save_path = os.path.join(WITH_MASK_DIR, f"{count_mask}.jpg")
            cv2.imwrite(save_path, face)
            count_mask += 1
        else:
            save_path = os.path.join(WITHOUT_MASK_DIR, f"{count_no_mask}.jpg")
            cv2.imwrite(save_path, face)
            count_no_mask += 1

print("Dataset converted!")
print("Mask:", count_mask)
print("No Mask:", count_no_mask)