import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report

from keras.callbacks import EarlyStopping

def train_model(train_gen, val_gen, model, epochs=10, early_stopping_patience=5, class_weights=None):
    """
    Trains a given model using the provided training and validation data generators.
    
    Parameters:
        - train_gen (Generator): The training data generator.
        - val_gen (Generator): The validation data generator.
        - model (tf.keras.Model): The model to be trained.
        - epochs (int, optional): The number of epochs to train the model, the default value is 10.
        - early_stopping_patience (int, optional): The number of epochs with no improvement (monitors 'val_loss') after which training will be stopped, the default value is 5.
        - class_weights (dict, optional): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function during training. Defaults to None.
    Returns:
        - model (tf.keras.Model): The trained model.
    """
    
    model.fit(
        train_gen,
        epochs=epochs,
        class_weight=class_weights,
        validation_data=val_gen,
        callbacks=[EarlyStopping(patience=early_stopping_patience, monitor='val_loss')]
    )
    
    return model

def get_classification_report(model, X_test, y_test):
    """
    Generates and prints a classification report for a given model and test data.

    Parameters:
        - model (object): The trained model used for making predictions.
        - X_test (array-like): The test data features.
        - y_test (array-like): The true labels for the test data.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))
    
    return None