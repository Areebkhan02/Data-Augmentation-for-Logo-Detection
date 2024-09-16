import os
import shutil

def combine_datasets(src1, src2, dest):
    # Create destination directories if they do not exist
    os.makedirs(os.path.join(dest, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'labels'), exist_ok=True)

    # Copy images from src1
    for file_name in os.listdir(os.path.join(src1, 'images')):
        full_file_name = os.path.join(src1, 'images', file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dest, 'images', file_name))

    # Copy labels from src1
    for file_name in os.listdir(os.path.join(src1, 'labels')):
        full_file_name = os.path.join(src1, 'labels', file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dest, 'labels', file_name))

    # Copy images from src2
    for file_name in os.listdir(os.path.join(src2, 'images')):
        full_file_name = os.path.join(src2, 'images', file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dest, 'images', file_name))

    # Copy labels from src2
    for file_name in os.listdir(os.path.join(src2, 'labels')):
        full_file_name = os.path.join(src2, 'labels', file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dest, 'labels', file_name))

# Define source and destination directories
src1 = 'Datasets/47_logos_dataset/10_classes_final/final/split2/train/combined_aug_org'
src2 = 'Datasets/47_logos_dataset/10_classes_final/final/split2/valid'
dest = 'Datasets/47_logos_dataset/10_classes_final/final/split2/train_comb'

# Combine datasets
combine_datasets(src1, src2, dest)

print('Datasets combined successfully!')
