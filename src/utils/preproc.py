import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

def resize_and_append(image_path, label, X, y, img_size):
    with Image.open(image_path) as img:
        img_resized = img.resize(img_size)
        img_array = np.array(img_resized)
        
        X.append(img_array)
        y.append(label)
        
def normalize_pixels(X_train, X_test, X_val):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_val = X_val / 255.0
    
    return X_train, X_test, X_val
    
def label_encode(y_train, y_test, y_val):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    y_val = le.fit_transform(y_val)
        
    return y_train, y_test, y_val

def load_and_preprocess_data(csv_path, desired_magnification, image_resolution, label_column):
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

def preproc_pipeline(desired_magnification, 
                     image_resolution, 
                     classification_type='binary'):

    csv_path = 'image_metadata/updated_image_data.csv'
    
    if classification_type == 'binary':
        label_column = 'Benign or Malignant'
    else: 
        classification_type = 'multiclass'
        label_column = 'Cancer Type'
    
    X_train, y_train, X_test, y_test, X_val, y_val = load_and_preprocess_data(csv_path, desired_magnification, image_resolution, label_column)
    
    # calculate class weights because our problem is unbalanced
    # np.unique makes this work for both binary (Benign/Malignant) and multiclass classification (Cancer Type)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: weight for i, weight in enumerate(class_weights)}
    
    datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    )
    
    # data augmentation generators
    # shuffles the data so no need to shuffle the data before passing it to the generator
    train_gen = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen = datagen.flow(X_val, y_val, batch_size=32, shuffle=True)
    
    return train_gen, val_gen, X_test, y_test, class_weights