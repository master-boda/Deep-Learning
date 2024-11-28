import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.utils import class_weight

from keras.callbacks import EarlyStopping

def train_model(X_train, y_train, X_val, y_val, model, epochs=10, batch_size=32, early_stopping_patience=3, data_gen=True):
    # class weights because our problem is unbalanced
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train) # np.unique(y_train) possibilitates usage for both binary and multiclass classification
    
    class_weights = {i: weight for i, weight in enumerate(class_weights)}
    
    # shuffle the data before training (it was not done in the preproc_pipeline function)
    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val = shuffle(X_val, y_val)

    model.fit(
        X_train, y_train,
        epochs=epochs,
        class_weight=class_weights,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=early_stopping_patience)])
        

    return model