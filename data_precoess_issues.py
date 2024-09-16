import os
from glob import glob
import shutil



# Define the paths
labels_path = "Datasets/47_logos_dataset/10_classes_final/final/split/train/aug_labels"
images_path = "Datasets/47_logos_dataset/10_classes_final/final/split/train/aug_images"

def check_bbox_issues(label_file):
    """
    Check if the bounding box coordinates are unnormalized, out of bounds, or negative.

    Args:
        label_file (str): Path to the label file.

    Returns:
        dict: Dictionary containing flags for unnormalized, out of bounds, or negative values.
    """
    issues = {'unnormalized': False, 'out_of_bounds': False, 'negative': False}
    
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            coords = list(map(float, line.split()[1:]))
            if any(coord > 1 for coord in coords):
                issues['unnormalized'] = True
            if any(coord < 0 for coord in coords):
                issues['negative'] = True
            if any(coord < 0 or coord > 1 for coord in coords):
                issues['out_of_bounds'] = True
    
    return issues

def process_labels(labels_path, images_path, delete=False):
    """
    Process the label files and optionally delete the associated images and labels with issues.

    Args:
        labels_path (str): Path to the folder containing label files.
        images_path (str): Path to the folder containing image files.
        delete (bool): Whether to delete files with issues.

    Returns:
        None
    """
    label_files = glob(os.path.join(labels_path, '*.txt'))
    total_files = len(label_files)
    unnormalized_count = 0
    out_of_bounds_count = 0
    negative_count = 0

    for label_file in label_files:
        issues = check_bbox_issues(label_file)
        
        if any(issues.values()):
            image_file = os.path.join(images_path, os.path.basename(label_file).replace('.txt', '.jpg'))

            if issues['unnormalized']:
                unnormalized_count += 1
            if issues['out_of_bounds']:
                out_of_bounds_count += 1
            if issues['negative']:
                negative_count += 1

            if delete:
                print(f"Deleting {label_file} and {image_file} due to issues: {issues}")
                os.remove(label_file)
                if os.path.exists(image_file):
                    os.remove(image_file)

    print(f"Total files processed: {total_files}")
    print(f"Unnormalized bounding boxes: {unnormalized_count}")
    print(f"Out of bounds bounding boxes: {out_of_bounds_count}")
    print(f"Negative bounding boxes: {negative_count}")

if __name__ == "__main__":
    process_labels(labels_path, images_path, delete=True)  # Set delete=False if you don't want to delete files
