import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
        
def update_image_paths(metadata):
    
    def get_image_location(row):
        
        possible_locations = ['train', 'test', 'val']
        for location in possible_locations:
            full_path = os.path.join('data', row['Magnification'], location, row['image_name'])
            if os.path.exists(full_path):
                return location
            
        print(f"WARNING: {row['image_name']} not found in any folder.")
        return "NOT FOUND"
    
    metadata.dropna(inplace=True) # Remove missing values (4 rows only)
    
    # These are temporary columns to help us find the new paths of the images
    metadata['image_name'] = metadata['path_to_image'].apply(lambda x: os.path.basename(x))
    metadata['image_location'] = metadata.apply(get_image_location, axis=1)
    
    # Update the paths to our new structure of folders
    metadata['path_to_image'] = metadata.apply(
        lambda row: os.path.join('data', row['Magnification'], row['image_location'], row['image_name']),
        axis=1
    )
    
    metadata.drop(columns=['image_name', 'image_location'], inplace=True)
    return metadata

if __name__ == '__main__':

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

    # The following function follows a similar thought process from practical class 15
    # The data (from OUTSIDE the repository) is moved to a data folder inside the repository and further split by magnification levels
    # The images for each magnification level is divided into "train", "test" and "val" folders
    # Note: the ".gitignore" file has instructions to ignore the "data" folder so that the repo. can be used without having to commit and push the ~4GB dataset
    def move_images(indices, metadata, target_directory, source_directory):
        """
        Move the images from the source directory to the target directory based on the indices.
        """
        
        for idx in indices:
            image_path = os.path.join(source_directory, metadata.iloc[idx]['path_to_image'])
            
            if os.path.exists(image_path): # Check if the file exists before trying to copy
                
                print(f"Copying {image_path} to {target_directory}")
                            
                # make sure that the image magnification level matches the magnification level of the target directory
                if metadata.iloc[idx]['Magnification'] != target_directory.split('\\')[-2]:
                    print(f"WARNING: Image magnification level does not match target directory: {image_path}, (target: {target_directory})")            
                
                # Create the destination folder if it doesn't exist
                os.makedirs(target_directory, exist_ok=True)
                
                shutil.copy2(image_path, target_directory) # Copy the image to the destination
            else:
                print(f"File not found: {image_path}")

    def setup_data(metadata_csv, source_directory):    
        
        metadata = pd.read_csv(metadata_csv)
        metadata = fix_csv(metadata) # Preprocessing of the metadata

        # Create combined labels for stratification
        metadata['combined_label'] = metadata['Benign or Malignant'] + '_' + metadata['Cancer Type']

        magnification_levels = metadata['Magnification'].unique()

        for magnification_level in magnification_levels:
            
            magnification_metadata = metadata[metadata['Magnification'] == magnification_level].reset_index(drop=True)
            indices = list(magnification_metadata.index)
            
            print(magnification_metadata['combined_label'].value_counts())  # Check the distribution

            train_directory = os.path.join('data', f'{magnification_level}', 'train')
            val_directory = os.path.join('data', f'{magnification_level}', 'val')
            test_directory = os.path.join('data', f'{magnification_level}', 'test')

            # Perform stratified split of images into train, val, and test sets
            train_indices, temp_indices = train_test_split(
                indices, train_size=0.7, random_state=42, stratify=magnification_metadata['combined_label']
            )
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=0.5, random_state=42, stratify=magnification_metadata.loc[temp_indices]['combined_label']
            )
            
            # Move train images to the train directory
            move_images(indices=train_indices, metadata=magnification_metadata, target_directory=train_directory, source_directory=source_directory)

            # Move val images to the val directory
            move_images(indices=val_indices, metadata=magnification_metadata, target_directory=val_directory, source_directory=source_directory)

            # Move test images to the test directory
            move_images(indices=test_indices, metadata=magnification_metadata, target_directory=test_directory, source_directory=source_directory)


    # Paths

    # DEFINE DIRECTORY PATH FOR DATASET (\DeepLearning24_25\)
    # ---------------------------------
    source_directory = r"D:\DeepLearning24_25"
    # ---------------------------------

    metadata_csv = os.path.join(source_directory, 'BreaKHis_v1 2/histology_slides/breast/image_data.csv')
    setup_data(metadata_csv, source_directory)

    # Update the paths in the metadata CSV to the new image locations
    img_metadata_df = pd.read_csv('image_metadata/image_data.csv')
    img_metadata_df = update_image_paths(img_metadata_df)
    img_metadata_df.to_csv('image_metadata/updated_image_data.csv', index=False)