import os
import cv2

# Path to your YOLO images
image_dirs = [
    r"C:\AI_Anti_Drone\datasets\visdrone\VisDrone2019-DET-train",
    r"C:\AI_Anti_Drone\datasets\visdrone\VisDrone2019-DET-val"
]

corrupt_images = []

for dir_path in image_dirs:
    for img_file in os.listdir(dir_path):
        if img_file.endswith((".jpg", ".png")):
            img_path = os.path.join(dir_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                corrupt_images.append(img_path)

if corrupt_images:
    print("Corrupt images found:")
    for path in corrupt_images:
        print(path)
else:
    print("All images are readable ✅")
