from keras.models import Model, Sequential, load_model
from keras.applications import VGG16, InceptionV3
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def binary_classification_vgg16_model(input_shape=(224, 224, 3), trainable_layers=10, learning_rate=0.0001):
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # train some layers and freeze others
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True

    model = Sequential()
    
    model.add(base_model)
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # low learning rate for fine tuning
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def multiclass_classification_vgg16_model(input_shape=(224, 224, 3), trainable_layers=10):
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # train some layers and freeze others
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True

    model = Sequential()
    
    model.add(base_model)
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(8, activation='softmax'))
    
    # low learning rate for fine tuning
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def sequential_multiclass_model(conv_filter_1=64,
								conv_filter_2=192,
								conv_filter_3=192,
								conv_filter_4=256,
								conv_filter_5=442,
								dense_units_0=512,
								dropout_rate_0=0.0,
								learning_rate=0.0001,
								input_shape=(224, 224, 3)):
	model = Sequential()
	model.add(Conv2D(conv_filter_1, (3, 3), activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(conv_filter_2, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(conv_filter_3, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(conv_filter_4, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(conv_filter_5, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Flatten())
	model.add(Dense(dense_units_0, activation='relu'))
	model.add(Dropout(dropout_rate_0))

	model.add(Dense(8, activation='softmax'))

	optimizer = Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	return model