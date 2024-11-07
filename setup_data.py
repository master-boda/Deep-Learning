import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

# C:\path\to\breakhis\histology_slides\breast
# Write your path to the directory where the dataset downloaded from Moodle is stored
source_directory = r"D:\DeepLearning24_25\BreaKHis_v1 2\histology_slides\breast"

train_directory = os.path.join('data', 'train')
test_directory = os.path.join('data', 'test')

os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

# Load the image metadata
metadata = pd.read_csv(os.path.join(source_directory, 'image_data.csv'))

# Remove missing values rows from the metadata (<0.1% of the dataset)
metadata = metadata.dropna()

# Create combined labels for stratification
metadata['combined_label'] = metadata['Benign or Malignant'] + '_' + metadata['Cancer Type']

# Create indices
indices = list(range(len(metadata)))

# Perform stratified split of images into train and test sets
train_indices, test_indices = train_test_split(
    indices, train_size=0.8, test_size=0.2, random_state=42, stratify=metadata['combined_label']
)

# Move train images to the train directory
for idx in train_indices:
    image_path = metadata.iloc[idx]['path_to_image']
    source_path = os.path.join(source_directory, image_path)
    destination_path = os.path.join(train_directory, image_path)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)

# Move test images to the test directory
for idx in test_indices:
    image_path = metadata.iloc[idx]['path_to_image']
    source_path = os.path.join(source_directory, image_path)
    destination_path = os.path.join(test_directory, image_path)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)