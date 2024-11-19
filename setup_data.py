import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

def fix_csv(metadata):
    """
    Fix the paths in the metadata CSV by replacing incorrect parts of the path
    
    Removes NaN rows in the metadata CSV.
    """
    metadata.dropna(inplace=True) # remove missing values
    
    # Replace the incorrect directory name with the correct one
    metadata['path_to_image'] = metadata['path_to_image'].apply(
        lambda x: x.replace('BreaKHis_v1/', 'BreaKHis_v1 2/')
    )
    
    return metadata

#The following function follows a similar thought process from practical class 15
# The data (from OUTSIDE the repository) is moved to a data folder inside the repository and further split by magnification levels
# The images for each magnification level is divided into "train", "test" and "val" folders
# Note: the ".gitignore" file has intructions to ignore the "data" folder so that the repo. can be used without having to commit and push the ~4GB dataset
def move_images(indices, metadata, target_directory, magnification_level=None):
    """
    Move the images from the source directory to the target directory based on the indices.
    """
    
    for idx in indices:
        image_path = os.path.join(source_directory, metadata.iloc[idx]['path_to_image'])
        
        if os.path.exists(image_path): # Check if the file exists before trying to copy

            print(f"Found file: {image_path}")
            
            # If image_path is /home/user/images/photo.jpg, the os.path.basename function will return photo.jpg.
            image_name = os.path.basename(image_path)
            
            if magnification_level:
                target_directory = os.path.join(target_directory, magnification_level)
            
            destination_path = os.path.join(target_directory, image_name)
            
            # Create the destination folder if it doesn't exist
            os.makedirs(target_directory, exist_ok=True)
            
            shutil.copy2(image_path, destination_path) # Copy the image to the destination
        else:
            print(f"File not found: {image_path}")

def move_images_by_magnification(indices, metadata, target_directory):
    """
    Move the images from the source directory to the target directory based on the indices and magnification levels.
    """
    magnification_levels = metadata['Magnification'].unique()
    
    for magnification_level in magnification_levels:
        magnification_indices = metadata[metadata['Magnification'] == magnification_level].index
        move_images(magnification_indices, metadata, target_directory, magnification_level)

def setup_data(train_directory, val_directory, test_directory, metadata_csv):
    
    metadata = pd.read_csv(metadata_csv)
    metadata = fix_csv(metadata) # Preprocessing of the metadata

    # Create combined labels for stratification
    metadata['combined_label'] = metadata['Benign or Malignant'] + '_' + metadata['Cancer Type']

    indices = list(range(len(metadata)))

    # Perform stratified split of images into train, val, and test sets
    train_indices, temp_indices = train_test_split(
        indices, train_size=0.7, random_state=42, stratify=metadata['combined_label']
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, stratify=metadata.iloc[temp_indices]['combined_label']
    )
    
    # Move train images to the train directory
    move_images(train_indices, metadata, train_directory)

    # Move val images to the val directory
    move_images(val_indices, metadata, val_directory)

    # Move test images to the test directory
    move_images(test_indices, metadata, test_directory)
    
    # Move images by magnification levels
    move_images_by_magnification(train_indices, metadata, train_directory)
    move_images_by_magnification(val_indices, metadata, val_directory)
    move_images_by_magnification(test_indices, metadata, test_directory)

source_directory = r"D:\DeepLearning24_25"

train_directory = os.path.join('data', 'train')
val_directory = os.path.join('data', 'val')
test_directory = os.path.join('data', 'test')
metadata_csv = os.path.join(source_directory, 'BreaKHis_v1 2/histology_slides/breast/image_data.csv')

setup_data(train_directory, val_directory, test_directory, metadata_csv)
