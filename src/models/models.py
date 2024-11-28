from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint

def binary_classification_baseline_model(input_shape):
    """  
    Parameters:
    input_shape (tuple): Shape of the input images.
    
    Returns:
    keras.Sequential: CNN model.
    """
    
    model = Sequential()
    
    model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def multiclass_classification_baseline_model(input_shape):
    """  
    Parameters:
    input_shape (tuple): Shape of the input images.
    
    Returns:
    keras.Sequential: CNN model.
    """
    
    model = Sequential()
    
    model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model