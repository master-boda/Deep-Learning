import pandas as pd
import numpy as np
from PIL import Image

def resize_and_append(image_path, label, X, y, img_size):
        with Image.open(image_path) as img:
            img_resized = img.resize(img_size)
            img_array = np.array(img_resized)
            
            X.append(img_array)
            y.append(label)

def preproc_pipeline(desired_magnification, 
                     image_resolution, 
                     csv_path='../image_metadata/updated_image_data.csv', 
                     classification_type=['binary','multiclass']):
    """
    Function to prepare data arrays for modeling.
    
    Parameters:
    csv_path (str): Path to the CSV file.
    desired_magnification (str): Desired magnification (e.g., '100X').
    image_resolution (int): Desired image resolution (e.g., 50).
    classification_type (str): Classification type, either 'binary' for 'Benign or Malignant' or 'multiclass' for 'Cancer Type'.
    
    Returns:
    tuple: Arrays for training, validation, and testing splits (X_train, y_train, X_test, y_test, X_val, y_val).
    """
    
    df = pd.read_csv(csv_path)
    
    # Filter the DataFrame based on the desired magnification
    df_filtered = df[df['Magnification'] == desired_magnification]
    
    if classification_type == 'binary':
        label_column = 'Benign or Malignant'
    elif classification_type == 'multiclass':
        label_column = 'Cancer Type'
    else:
        raise ValueError("classification_type must be either 'binary' or 'multiclass'")
    
    # Create lists to store the arrays
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_val, y_val = [], []
    
    # Resize images and append to lists

    for _, row in df_filtered.iterrows():
        image_path = row['path_to_image']
        label = row[label_column]
        if 'train' in image_path:
            resize_and_append(image_path, label, X_train, y_train, image_resolution)
        elif 'test' in image_path:
            resize_and_append(image_path, label, X_test, y_test, image_resolution)
        elif 'val' in image_path:
            resize_and_append(image_path, label, X_val, y_val, image_resolution)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

# Example usage:
# csv_path = 'path_to_your_csv_file.csv'
# desired_magnification = '100X'
# image_resolution = 50
# classification_type = 'binary'  # or 'multiclass'
# X_train, y_train, X_test, y_test, X_val, y_val = preproc_pipeline(desired_magnification='40X', image_resolution=200, classification_type='multiclass')
# X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape