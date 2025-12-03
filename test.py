import os

splits = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val']
base_path = 'datasets/visdrone'

for split in splits:
    print("="*50)
    print(f"Checking {split}:")
    split_path = os.path.join(base_path, split)
    images_path = os.path.join(split_path, 'images')
    annotations_path = os.path.join(split_path, 'annotations')
    labels_path = os.path.join(split_path, 'labels_fixed')

    # Images
    if os.path.exists(images_path):
        img_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png'))]
        print(f"  Images folder: {len(img_files)} files found")
        if len(img_files) == 0:
            print("  WARNING: No images found!")
    else:
        print(f"  ERROR: Images folder missing at {images_path}")

    # Annotations
    if os.path.exists(annotations_path):
        ann_files = [f for f in os.listdir(annotations_path) if f.lower().endswith(('.txt', '.xml'))]
        print(f"  Annotations folder: {len(ann_files)} files found")
        if len(ann_files) == 0:
            print("  WARNING: No annotation files found!")
    else:
        print(f"  ERROR: Annotations folder missing at {annotations_path}")

    # Labels (YOLO format, optional)
    if os.path.exists(labels_path):
        lbl_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]
        print(f"  Labels_fixed folder: {len(lbl_files)} files found")
        if len(lbl_files) == 0:
            print("  NOTE: No YOLO label files found!")
    else:
        print(f"  NOTE: labels_fixed folder not present at {labels_path}")
    print("="*50)

print("VisDrone dataset verification complete.")
