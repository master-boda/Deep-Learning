import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score

def train_model(X_train, y_train, X_val, y_val, model, epochs=10, batch_size=32, early_stopping_patience=4):
    # Class weights because our problem is unbalanced
    class_weights = class_weight.compute_class_weight('balanced', 
                                                      classes=np.unique(y_train), 
                                                      y=y_train)
    
    class_weights = {i: weight for i, weight in enumerate(class_weights)}
    
    X_train, y_train = shuffle(X_train, y_train)

    model.fit(X_train, y_train, epochs=epochs, class_weight=class_weights, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[EarlyStopping(patience=early_stopping_patience)])

    return model