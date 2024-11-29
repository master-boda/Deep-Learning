import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.utils import class_weight

from keras.callbacks import EarlyStopping

def train_model(train_gen, val_gen, model, epochs=10, early_stopping_patience=3, class_weights=None):
    model.fit(
        train_gen,
        epochs=epochs,
        class_weight=class_weights,
        validation_data=val_gen,
        callbacks=[EarlyStopping(patience=early_stopping_patience, monitor='val_loss')]
    )
    
    return model