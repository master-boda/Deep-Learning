import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

def resize_and_append(image_path, label, X, y, img_size):
    """
    Resize an image and append it along with its label to the provided lists.
    
    Parameters:
        - image_path (str): The file path to the image to be resized.
        - label (any): The label associated with the image.
        - X (list): The list to which the resized image array will be appended.
        - y (list): The list to which the label will be appended.
        - img_size (tuple): The target size for resizing the image (width, height).
        
    Returns:
        None
    """
    with Image.open(image_path) as img:
        img_resized = img.resize(img_size)
        img_array = np.array(img_resized)
        
        X.append(img_array)
        y.append(label)
        
def normalize_pixels(X_train, X_test, X_val):
    """
    Normalize pixel values of training, testing, and validation datasets.
    Scales the pixel values of the input datasets to the range [0, 1]
    by dividing each pixel value by 255.0 (max pixel intensity).
    
    Parameters:
        - X_train (numpy.ndarray): The training dataset with pixel values.
        - X_test (numpy.ndarray): The testing dataset with pixel values.
        - X_val (numpy.ndarray): The validation dataset with pixel values.
        
    Returns:
        - X_train (numpy.ndarray): Normalized training set images.
        - X_test (numpy.ndarray): Normalized testing set images.
        - X_val (numpy.ndarray): Normalized validation set images.
    """
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_val = X_val / 255.0
    
    return X_train, X_test, X_val
    
def label_encode(y_train, y_test, y_val):
    """
    Encodes the labels of the training, testing, and validation datasets using label encoding.
    
    Parameters:
        - y_train (array-like): The labels for the training dataset.
        - y_test (array-like): The labels for the testing dataset.
        - y_val (array-like): The labels for the validation dataset.
        
    Returns:
        - y_train (numpy.ndarray): Encoded training set labels.
        - y_test (numpy.ndarray): Encoded testing set labels.
        - y_val (numpy.ndarray): Encoded validation set labels.
    """
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    y_val = le.fit_transform(y_val)
        
    return y_train, y_test, y_val

def load_and_preprocess_data(csv_path, desired_magnification, image_resolution, label_column):
    """
    Load and preprocess image data from the CSV file containing the image metadata.
    This function reads image data paths and labels from the CSV file, filters the data based on the desired magnification,
    resizes the images to the specified resolution, sorts the data into training, testing, and validation arrays.
    The images are then normalized, and the labels are encoded.
    
    Parameters:
        - csv_path (str): Path to the image metadata CSV.
        - desired_magnification (int): The magnification level to filter the images (40X, 100X, 200X, 400X).
        - image_resolution (tuple): The desired resolution to resize the images (width, height).
        - label_column (str): The name of the column in the CSV file that contains the labels ('Benign or Malignant' or 'Cancer Type').
        
    Returns:
        - X_train (numpy.ndarray): Training set images.
        - y_train (numpy.ndarray): Training set labels.
        - X_test (numpy.ndarray): Testing set images.
        - y_test (numpy.ndarray): Testing set labels.
        - X_val (numpy.ndarray): Validation set images.
        - y_val (numpy.ndarray): Validation set labels.
    """
    df = pd.read_csv(csv_path)
    
    # select only the rows for the selected magnification (40X, 100X, 200X, 400X)
    df_filtered = df[df['Magnification'] == desired_magnification]
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_val, y_val = [], []
    
    # it is necessary to use the updated_image_data.csv file to get the correct path to the images
    for boda, row in df_filtered.iterrows():
        image_path = row['path_to_image']
        label = row[label_column]
        if 'train' in image_path:
            resize_and_append(image_path, label, X_train, y_train, image_resolution)
        elif 'test' in image_path:
            resize_and_append(image_path, label, X_test, y_test, image_resolution)
        elif 'val' in image_path:
            resize_and_append(image_path, label, X_val, y_val, image_resolution)
            
    # convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # label encode the target variable (use sparse_categorical_crossentropy as loss function for multiclass)
    y_train, y_test, y_val = label_encode(y_train, y_test, y_val)
    
    # normalize pixel intensities
    X_train, X_test, X_val = normalize_pixels(X_train, X_test, X_val)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def data_augmentation(X_train, y_train, datagen, augmented_images_per_image):
    """
    Perform data augmentation on the training dataset.
    
    Parameters:
        - X_train (numpy.ndarray): Array of training images.
        - y_train (numpy.ndarray): Array of training labels.
        - datagen (ImageDataGenerator): Keras ImageDataGenerator instance for generating augmented images.
        - augmented_images_per_image (int): Number of augmented images to generate per original image.
        
    Returns:
        - X_train_augmented (numpy.ndarray): Array of augmented training images.
        - y_train_augmented (numpy.ndarray): Array of augmented training labels.
    """
    augmented_images = []
    augmented_labels = []

    # also include original images in the data augmentation
    for i in range(len(X_train)):
        augmented_images.append(X_train[i])
        augmented_labels.append(y_train[i])

    # generate augmented images
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        x = x.reshape((1,) + x.shape) # datagen.flow expects 4D arrays, so we need to reshape the 3D array
        boda = 0
        for batch in datagen.flow(x, batch_size=1): # generate 1 augmented image per iteration
            augmented_images.append(batch[0])
            augmented_labels.append(y)
            boda += 1
            if boda >= augmented_images_per_image:
                break

    X_train_augmented = np.array(augmented_images)
    y_train_augmented = np.array(augmented_labels)
    
    return X_train_augmented, y_train_augmented

def preproc_pipeline(desired_magnification, 
                     image_resolution, 
                     classification_type='binary',
                     use_data_augmentation=False,
                     augmented_images_per_image=5,
                     csv_path = 'image_metadata/updated_image_data.csv',
                     batch_size=32):
    """
    Preprocess image data.

    This function loads image data from the CSV file containing image metadata, filters it based on the desired magnification,
    resizes the images, normalizes pixel values, encodes labels, and optionally performs data augmentation on the training set.
    It returns data generators for training and validation, as well as the test dataset and class weights.

    Parameters:
        - desired_magnification (int): The magnification level to filter the images (40X, 100X, 200X, 400X).
        - image_resolution (tuple): The desired resolution to resize the images (width, height).
        - classification_type (str, optional): The type of classification ('binary' or 'multiclass'). Defaults to 'binary'.
        - use_data_augmentation (bool, optional): Whether to perform data augmentation on the training dataset. Defaults to False.
        - augmented_images_per_image (int, optional): Number of augmented images to generate per original image. Defaults to 5.
        - csv_path (str, optional): Path to the image metadata CSV file. Defaults to 'image_metadata/updated_image_data.csv'.
        - batch_size (int, optional): The batch size for the data generators. Defaults to 32.

    Returns:
        - train_gen (Iterator): Data generator for the training dataset.
        - val_gen (Iterator): Data generator for the validation dataset.
        - X_test (numpy.ndarray): Array of test set images.
        - y_test (numpy.ndarray): Array of test set labels.
        - class_weights (dict): Dictionary of class weights to handle class imbalance.
    """
    
    if classification_type == 'binary':
        label_column = 'Benign or Malignant'
    else: 
        classification_type = 'multiclass'
        label_column = 'Cancer Type'
    
    X_train, y_train, X_test, y_test, X_val, y_val = load_and_preprocess_data(csv_path, desired_magnification, image_resolution, label_column)
    
    datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    )
    
    if use_data_augmentation == True:
        print(f'Number of training images before data augmentation: {len(X_train)}')
        X_train, y_train = data_augmentation(X_train, y_train, datagen, augmented_images_per_image)
        print(f'Number of training images after data augmentation: {len(X_train)}')
    
    # calculate class weights because our problem is unbalanced
    # np.unique makes this work for both binary (Benign/Malignant) and multiclass classification (Cancer Type)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: weight for i, weight in enumerate(class_weights)}

    
    # data augmentation generators
    # shuffles the data so no need to shuffle the data before passing it to the generator
    train_gen = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=True)
    
    return train_gen, val_gen, X_test, y_test, class_weights