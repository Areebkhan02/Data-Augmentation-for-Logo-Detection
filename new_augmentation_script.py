import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Define augmentation pipelines with different rotations
augmentation_pipelines = [
    A.Compose([
        A.Rotate(limit=(45, 180), p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0)),

    # A.Compose([
    #     A.Rotate(limit=(0, 45), p=0.2),
    #     A.RandomBrightnessContrast(p=0.3),
    #     A.HueSaturationValue(p=0.2)
    # ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0)),

    # A.Compose([
    #     A.Rotate(limit=(180, 270), p=1.0),
    #     A.RandomBrightnessContrast(p=0.7),
    #     A.HueSaturationValue(p=0.7)
    # ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0))
]

# Separate pipeline for the 'more' label
more_augmentation_pipeline = A.Compose([
    A.Rotate(limit=180, p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0))

# Helper function to load image and annotations
def load_data(image_path, annotation_path):
    img = cv2.imread(image_path)
    with open(annotation_path, 'r') as f:
        bboxes = []
        class_labels = []
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return img, {'class_labels': class_labels, 'bboxes': bboxes}

# Helper function to save augmented image and annotation
def save_data(img, ann, filename, image_dir, annotation_dir):
    img_path = os.path.join(image_dir, filename)
    ann_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
    cv2.imwrite(img_path, img)
    with open(ann_path, 'w') as f:
        for class_id, bbox in zip(ann['class_labels'], ann['bboxes']):
            line = f"{class_id} {' '.join(map(str, bbox))}\n"
            f.write(line)

# Normalize bounding boxes to ensure they are within [0.0, 1.0]
def normalize_bboxes(bboxes, img_width, img_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        height = np.clip(height, 0.0, 1.0)
        normalized_bboxes.append([x_center, y_center, width, height])
    return normalized_bboxes

# Function to augment a single image and annotation
def augment_single_image(img, ann, filename, minority_labels, medium_labels, more_labels, image_dir, annotation_dir):
    class_labels = ann['class_labels']
    bboxes = ann['bboxes']

    # Normalize bounding boxes
    img_height, img_width = img.shape[:2]
    bboxes = normalize_bboxes(bboxes, img_width, img_height)

    # Augment using different pipelines based on label type
    if any(cls in minority_labels for cls in class_labels):
        for i in range(augmentation_count_minority):
            for idx, aug_pipeline in enumerate(augmentation_pipelines):
                try:
                    augmented = aug_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                    aug_filename = f'aug_minority_{idx}_{i}_{filename}'
                    save_data(augmented['image'], {
                        'class_labels': augmented['class_labels'],
                        'bboxes': augmented['bboxes']
                    }, aug_filename, image_dir, annotation_dir)
                except ValueError as e:
                    print(f"Error augmenting {filename}: {e}")
                    continue

    elif any(cls in medium_labels for cls in class_labels):
        for idx, aug_pipeline in enumerate(augmentation_pipelines):
            try:
                augmented = aug_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_filename = f'aug_medium_{idx}_{filename}'
                save_data(augmented['image'], {
                    'class_labels': augmented['class_labels'],
                    'bboxes': augmented['bboxes']
                }, aug_filename, image_dir, annotation_dir)
            except ValueError as e:
                print(f"Error augmenting {filename}: {e}")
                continue

    elif any(cls in more_labels for cls in class_labels):
        try:
            augmented = more_augmentation_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_filename = f'aug_more_{filename}'
            save_data(augmented['image'], {
                'class_labels': augmented['class_labels'],
                'bboxes': augmented['bboxes']
            }, aug_filename, image_dir, annotation_dir)
        except ValueError as e:
            print(f"Error augmenting {filename}: {e}")

# Define directories
image_dir = 'Datasets/47_logos_dataset/10_classes_final/final/split/train/images'
annotation_dir = 'Datasets/47_logos_dataset/10_classes_final/final/split/train/labels'
augmented_image_dir = 'Datasets/47_logos_dataset/10_classes_final/final/split/train/aug_images'
augmented_annotation_dir = 'Datasets/47_logos_dataset/10_classes_final/final/split/train/aug_labels'

# Create directories to save augmented images and annotations if they don't exist
os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_annotation_dir, exist_ok=True)

# Define labels
minority_labels = []  # Example class IDs for minority classes
medium_labels = [30]  # Example class IDs for medium classes
more_labels = []  # Example class IDs for more classes

# Define augmentation counts
augmentation_count_minority = 2  # Number of times to augment minority class images


# Process each image and annotation
for filename in tqdm(os.listdir(image_dir), desc="Processing images"):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_dir, filename)
        ann_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
        if os.path.exists(ann_path):
            img, ann = load_data(img_path, ann_path)
            augment_single_image(img, ann, filename, minority_labels, medium_labels, more_labels, augmented_image_dir, augmented_annotation_dir)
