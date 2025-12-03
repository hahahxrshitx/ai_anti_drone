import os

# Base dataset path
base_dir = r"C:\AI_Anti_Drone\datasets"

# Define splits
splits = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val"
}

# Valid VisDrone classes (1–10)
valid_classes = list(range(1, 11))  # pedestrian → motor

# Image size (VisDrone images are 960x540)
IMG_W, IMG_H = 960.0, 540.0

for split, folder in splits.items():
    dataset_dir = os.path.join(base_dir, folder)
    img_dir = os.path.join(dataset_dir, "images")
    ann_dir = os.path.join(dataset_dir, "annotations")

    # ✅ New output folder so old labels remain safe
    label_dir = os.path.join(dataset_dir, "labels_fixed")
    os.makedirs(label_dir, exist_ok=True)

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".txt"):
            continue

        ann_path = os.path.join(ann_dir, ann_file)
        label_path = os.path.join(label_dir, ann_file)

        with open(ann_path, "r") as f:
            lines = f.readlines()

        label_lines = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 6:  # skip malformed rows
                continue

            try:
                x, y, w, h = map(int, parts[:4])
                cls_id = int(parts[5])
            except ValueError:
                # Skip bad lines safely
                continue

            # Skip invalid classes
            if cls_id not in valid_classes:
                continue

            # Convert to YOLO format
            x_center = (x + w / 2) / IMG_W
            y_center = (y + h / 2) / IMG_H
            width = w / IMG_W
            height = h / IMG_H

            # YOLO class IDs start from 0
            label_line = f"{cls_id-1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            label_lines.append(label_line)

        # Save converted labels
        with open(label_path, "w") as f:
            f.writelines(label_lines)

    print(f"✅ {split} conversion done! Labels saved to {label_dir}")
