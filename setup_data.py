import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
        
def update_image_paths(metadata):
    """
    Update the image paths in the metadata DataFrame to match the new folder structure.
    This function updates the 'path_to_image' column in the metadata DataFrame by determining
    the new location of each image based on whether it is in the 'train',
    'test', or 'val' folder. If an image is not found in any of these folders, a warning is printed
    and the image is marked as "NOT FOUND".
    
    Parameters:
        - metadata (pd.DataFrame): A DataFrame containing image metadata, including a 'path_to_image' column
                             with the original paths to the images.
                             
    Returns:
        - metadata (pd.DataFrame): The updated metadata DataFrame with the 'path_to_image' column using the new
                        folder structure.
    """
    
    def get_image_location(row):
        
        possible_locations = ['train', 'test', 'val']
        for location in possible_locations:
            full_path = os.path.join('data', location, row['image_name'])
            if os.path.exists(full_path):
                return location
            
        print(f"WARNING: {row['image_name']} not found in any folder.")
        return "NOT FOUND"
    
    metadata.dropna(inplace=True) # remove missing values (4 rows only)
    
    # these are temporary columns to help us find the new paths of the images
    metadata['image_name'] = metadata['path_to_image'].apply(lambda x: os.path.basename(x))
    metadata['image_location'] = metadata.apply(get_image_location, axis=1)
    
    # update the paths to our new structure of folders
    metadata['path_to_image'] = metadata.apply(
        lambda row: os.path.join('data', row['image_location'], row['image_name']),
        axis=1
    )
    
    metadata.drop(columns=['image_name', 'image_location'], inplace=True)
    
    return metadata

if __name__ == '__main__':  # avoid running the code when importing the module

    def fix_csv(metadata):
        """
        Cleans and fixes the paths in the given metadata DataFrame.
        This function performs the following operations on the input DataFrame:
        1. Removes rows with missing values.
        2. Replaces incorrect directory names in the 'path_to_image' column with the
        correct ones (for some reason the downloaded metadata from Moodle contains a " 2"
        in the 'BreaKHis_v1/' path section).
        
        Parameters:
            - metadata (pandas.DataFrame): The input DataFrame containing metadata with a 'path_to_image' column.
            
        Returns:
            - metadata (pandas.DataFrame): The cleaned and updated DataFrame.
        """
        metadata.dropna(inplace=True) # remove missing values

        # replace the incorrect directory name with the correct one
        metadata['path_to_image'] = metadata['path_to_image'].apply(
            lambda x: x.replace('BreaKHis_v1/', 'BreaKHis_v1 2/')
        )
        
        return metadata

    # the following function follows a similar thought process from practical class 15
    # the data (from OUTSIDE the repository) is moved to a data folder inside the repository and further split by magnification levels
    # the images for each magnification level is divided into "train", "test" and "val" folders
    # note: the ".gitignore" file has instructions to ignore the "data" folder so that the repo. can be used without having to commit and push the ~4GB dataset
    def move_images(indices, metadata, target_directory, source_directory):
        """
        Moves images from the source directory to the target directory based on the provided indices and metadata.
        
        Parameters:
            - indices (list): List of indices indicating which images to move.
            - metadata (pandas.DataFrame): DataFrame containing metadata about the images, including 'path_to_image' and 'Magnification' columns.
            - target_directory (str): The directory where the images should be moved to.
            - source_directory (str): The directory where the images are currently located.
            
        Returns:
            None
            
        Notes:
            - The function checks if the image file exists before attempting to copy it.
            - It prints a warning if the image magnification level does not match the magnification level of the target directory.
            - It creates the target directory if it does not exist.
            - It prints a message if the file is not found in the source directory.
        """
        
        for idx in indices:
            image_path = os.path.join(source_directory, metadata.loc[idx, 'path_to_image'])
            
            if os.path.exists(image_path): # check if the file exists before trying to copy
                
                print(f"Copying {image_path} to {target_directory}")
                            
                # create the destination folder if it doesn't exist
                os.makedirs(target_directory, exist_ok=True)
                
                shutil.copy2(image_path, target_directory) # copy the image to the destination
            else:
                print(f"File not found: {image_path}")

    def setup_data(metadata_csv, source_directory):    
        """
        Sets up the data by reading metadata from the image metadata CSV file, preprocessing it, and performing a stratified split of images 
        into training, validation, and test sets based on combined labels.
        
        Parameters:
            - metadata_csv (str): Path to the CSV file containing metadata.
            - source_directory (str): Path to the directory containing the source images.
            
        Returns:
            None
            
        The function performs the following steps:
        1. Reads the image metadata from the CSV file.
        2. Preprocesses the metadata using the `fix_csv` function.
        3. Creates combined labels for stratification by concatenating 'Benign or Malignant' and 'Cancer Type' columns.
        4. Filters the metadata.
        5. Prints the distribution of combined labels.
        6. Creates directories for training, validation, and test sets.
        7. Performs a stratified split of images into training, validation, and test sets.
        8. Moves the images to their respective directories.
        """
        
        metadata = pd.read_csv(metadata_csv)
        metadata = fix_csv(metadata) # preprocessing of the metadata

        # create combined labels for stratification
        metadata['combined_label'] = metadata['Benign or Malignant'] + '_' + metadata['Cancer Type']
        metadata.reset_index(drop=True, inplace=True)

        indices = list(metadata.index)
        
        print(metadata['combined_label'].value_counts())  # check the distribution

        train_directory = os.path.join('data', 'train')
        val_directory = os.path.join('data', 'val')
        test_directory = os.path.join('data', 'test')

        # perform stratified split of images into train, val, and test sets
        train_indices, temp_indices = train_test_split(
            indices, train_size=0.7, random_state=42, stratify=metadata['combined_label']
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42, stratify=metadata.loc[temp_indices]['combined_label']
        )
        
        # move train images to the train directory
        move_images(indices=train_indices, metadata=metadata, target_directory=train_directory, source_directory=source_directory)

        # move val images to the val directory
        move_images(indices=val_indices, metadata=metadata, target_directory=val_directory, source_directory=source_directory)

        # move test images to the test directory
        move_images(indices=test_indices, metadata=metadata, target_directory=test_directory, source_directory=source_directory)


    # paths

    # define directory path for dataset (\DeepLearning24_25\)
    # ---------------------------------
    source_directory = r"D:\DeepLearning24_25"
    # ---------------------------------

    metadata_csv = os.path.join(source_directory, 'BreaKHis_v1 2/histology_slides/breast/image_data.csv')
    setup_data(metadata_csv, source_directory)

    # update the paths in the metadata CSV to the new image locations
    img_metadata_df = pd.read_csv('image_metadata/image_data.csv')
    img_metadata_df = update_image_paths(img_metadata_df)
    img_metadata_df.to_csv('image_metadata/updated_image_data.csv', index=False)
