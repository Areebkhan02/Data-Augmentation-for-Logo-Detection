import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Define augmentation pipelines with different rotations
augmentation_pipelines = [
    A.Compose([
        A.Rotate(limit=(45, 180), p=1.0),
        A.RandomBrightnessContrast(p=0.8),
        A.HueSaturationValue(p=0.8)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0)),

    A.Compose([
        A.Rotate(limit=(-95, -45), p=1.0),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0)),

    A.Compose([
        A.Rotate(limit=(180, 270), p=1.0),
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.7)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0))
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

# Function to augment based on label priority
def augment_based_on_labels(img, ann, filename, minority_classes, medium_classes, augmentation_count, image_dir, annotation_dir):
    class_labels = ann['class_labels']
    class_labels1 = set(ann['class_labels'])
    #print(class_labels1)
    
    if class_labels1.intersection(minority_classes):
        pipelines = augmentation_pipelines
        runs = augmentation_count
        sec = "minority"
    elif class_labels1.intersection(medium_classes):
        pipelines = more_augmentation_pipeline
        runs = 1
        sec = "medium"
    else:
        # pipelines = [more_augmentation_pipeline]
        # runs = 1
        # sec = "more"
        return

    
    # Normalize bounding boxes
    img_height, img_width = img.shape[:2]
    bboxes = normalize_bboxes(ann['bboxes'], img_width, img_height)
    
    for i in range(runs):
        for j, pipeline in enumerate(pipelines):
            try:
                augmented = pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_filename = f'aug_{sec}_{i}_{j}_{filename}'
                save_data(augmented['image'], {
                    'class_labels': augmented['class_labels'],
                    'bboxes': augmented['bboxes']
                }, aug_filename, image_dir, annotation_dir)
            except ValueError as e:
                print(f"Error augmenting {filename}: {e}")
                continue

# Define directories
image_dir = 'Atheritia/Datasets/47_logos_dataset/final_47_logos_dataset/train/images'
annotation_dir = 'Atheritia/Datasets/47_logos_dataset/final_47_logos_dataset/train/labels'
augmented_image_dir = 'Atheritia/Datasets/47_logos_dataset/final_47_logos_dataset/train/aug_images'
augmented_annotation_dir = 'Atheritia/Datasets/47_logos_dataset/final_47_logos_dataset/train/aug_labels'

# Create directories to save augmented images and annotations if they don't exist
os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_annotation_dir, exist_ok=True)

# Define minority, medium, and more classes
minority_classes = {43,52}
medium_classes = {49,26,45,20,62,59,19,33,34,57,63,39,18,32,53,48,31,37,41,28}
more_classes = {}

# Define augmentation count for minority classes
augmentation_count = 2

# Process each image and annotation
for filename in tqdm(os.listdir(image_dir), desc="Processing images"):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_dir, filename)
        ann_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
        if os.path.exists(ann_path):
            img, ann = load_data(img_path, ann_path)
            augment_based_on_labels(img, ann, filename, minority_classes, medium_classes, augmentation_count, augmented_image_dir, augmented_annotation_dir)
