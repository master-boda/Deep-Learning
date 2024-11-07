import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

# The CSV file containing the image metadata contains a column that stores the images paths
# The paths however dont match the ones in Moodle due to a file name difference ('BreaKHis_v1/', 'BreaKHis_v1 2/')

# As seen in the data exploration notebook there is an extremely small amount of missing values, (<0.1% of the dataset)
# These rows are dropped

# These two tasks explained above are done in the following function
def fix_csv(metadata):
    """
    Fix the paths in the metadata CSV by replacing incorrect parts of the path
    
    Removes NaN rows in the metadata CSV.
    """
    # remove missing values
    metadata.dropna(inplace=True)
    
    # Replace the incorrect directory name with the correct one
    metadata['path_to_image'] = metadata['path_to_image'].apply(
        lambda x: x.replace('BreaKHis_v1/', 'BreaKHis_v1 2/')
    )
    
    return metadata

def move_images(indices, metadata, target_directory):
    """
    Move the images from the source directory to the target directory based on the indices.
    """
    for idx in indices:
        image_path = os.path.join(source_directory, metadata.iloc[idx]['path_to_image'])
        
        # Check if the file exists before trying to copy
        if os.path.exists(image_path):
            print(f"Found file: {image_path}")
            
            # Get just the image filename
            image_name = os.path.basename(image_path)
            destination_path = os.path.join(target_directory, image_name)
            
            # Create the destination folder if it doesn't exist
            os.makedirs(target_directory, exist_ok=True)
            
            # Copy the image to the destination
            shutil.copy2(image_path, destination_path)
        else:
            print(f"File not found: {image_path}")

# Main function to set up the data
def setup_data(train_directory, test_directory, metadata_csv):
    # Load the image metadata
    metadata = pd.read_csv(metadata_csv)
    
    # Fix the paths in the metadata and remove NaN rows
    metadata = fix_csv(metadata)

    # Create combined labels for stratification
    metadata['combined_label'] = metadata['Benign or Malignant'] + '_' + metadata['Cancer Type']

    # Create indices
    indices = list(range(len(metadata)))

    # Perform stratified split of images into train and test sets
    train_indices, test_indices = train_test_split(
        indices, train_size=0.8, test_size=0.2, random_state=42, stratify=metadata['combined_label']
    )
    
    # Move train images to the train directory
    move_images(train_indices, metadata, train_directory)

    # Move test images to the test directory
    move_images(test_indices, metadata, test_directory)

# Paths

# DEFINE DIRECTORY PATH FOR DATASET (\DeepLearning24_25\BreaKHis_v1 2\histology_slides\breast)
# ---------------------------------
source_directory = r"D:\DeepLearning24_25"
# ---------------------------------

train_directory = os.path.join('data', 'train')
test_directory = os.path.join('data', 'test')
metadata_csv = os.path.join(source_directory, 'BreaKHis_v1 2/histology_slides/breast/image_data.csv')

# Setup the data
setup_data(train_directory, test_directory, metadata_csv)
