import cv2
import numpy as np
import os

image_folder = 'datasets/visdrone/VisDrone2019-DET-val/images'
attacks = ['blur', 'noise', 'occlusion', 'dark', 'bright']

for attack in attacks:
    os.makedirs(f'adversarial_images/{attack}', exist_ok=True)

images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
for img_file in images:
    img_path = os.path.join(image_folder, img_file)
    img = cv2.imread(img_path)
    # BLUR
    blur = cv2.GaussianBlur(img, (9,9), 1.5)
    cv2.imwrite(f'adversarial_images/blur/{img_file}', blur)
    # NOISE
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    cv2.imwrite(f'adversarial_images/noise/{img_file}', noisy)
    # OCCLUSION
    occ_img = img.copy()
    h, w = occ_img.shape[:2]
    occ_img[0:h//4, 0:w//4] = 0
    cv2.imwrite(f'adversarial_images/occlusion/{img_file}', occ_img)
    # DARK
    dark = cv2.convertScaleAbs(img, alpha=0.5, beta=-50)
    cv2.imwrite(f'adversarial_images/dark/{img_file}', dark)
    # BRIGHT
    bright = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
    cv2.imwrite(f'adversarial_images/bright/{img_file}', bright)
