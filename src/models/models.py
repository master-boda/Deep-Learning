from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.applications import VGG16
from keras.optimizers import Adam

def binary_classification_baseline_model(input_shape):
    
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

def binary_classification_vgg16_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model