import numpy as np
import tensorflow as tf
import os
import json
import pickle

from src.utils.preproc import *
from src.utils.visualizations import plot_confusion_matrix

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from keras.models import Model, Sequential, load_model
from keras.applications import VGG16, InceptionV3
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tabulate import tabulate

def train_model(train_gen, val_gen, model, callbacks, epochs=10, class_weights=None, steps_per_epoch=None):
    """
    Trains a given model using the provided training and validation data generators.
    
    Parameters:
        - train_gen (Generator): The training data generator.
        - val_gen (Generator): The validation data generator.
        - model (tf.keras.Model): The model to be trained.
        - epochs (int, optional): The number of epochs to train the model, the default value is 10.
        - class_weights (dict, optional): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function during training. Defaults to None.
        - steps_per_epoch (int, optional): The number of steps (batches of samples) to yield from the generator before declaring one epoch finished and starting the next epoch. Defaults to None.
        
    Returns:
        - model (tf.keras.Model): The trained model.
    """
    
    model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weights,
        validation_data=val_gen,
        callbacks=callbacks)
    
    return model

def save_model(model, path="src/models", model_name="saved_model.h5", history_name="training_history.json"):
    """
    Save the TensorFlow model and its training history to the specified path.
    
    Parameters:
    - model: TensorFlow model to be saved.
    - history: Training history to be saved.
    - path: Destination path where the model and history will be saved. Default is "src/models".
    - model_name: Name of the model file. Default is "saved_model.h5".
    - history_name: Name of the history file. Default is "training_history.json".
    
    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    model.save(os.path.join(path, model_name), save_format='h5')
    
def save_training_history(history, path, history_name="model_history.pkl"):
    """
    Save the training history to the specified path using pickle.
        
    Parameters:
    - history: Training history to be saved.
    - path: Destination path where the history will be saved. Default is "src/models".
    - history_name: Name of the history file. Default is "training_history.pkl".
        
    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    with open(os.path.join(path, history_name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def evaluate_model(model, classification_type='binary', show_confusion_matrix=True):
    """
    Evaluate a TensorFlow model and plot evaluation metrics.
    
    Parameters:
        - model (tf.keras.Model): The trained TensorFlow model.
        - test_generator (Generator): The test data generator.
        - classification_type (str): Type of classification ('binary' or 'multiclass').
        
    Returns:
        - results (dict): Dictionary containing evaluation metrics.
    """
    print("Loading test data...")
    train_gen, val_gen, test_gen, class_weights, steps = preproc_pipeline((224, 224), classification_type=classification_type, use_data_augmentation=False)
    
    print("Starting model evaluation...")
    
    # reset the generator before making predictions
    test_gen.reset()
    
    total_samples = test_gen.n
    print(f"Total number of samples in the test generator: {total_samples}")
    
    y_pred_prob = model.predict_generator(test_gen, steps=(total_samples // test_gen.batch_size) + 1) # +1 to account for remaining samples that don't make a full batch
    
    # handles both binary and multiclass classification
    if classification_type == 'binary':
        y_pred = (y_pred_prob > 0.5).astype(int).flatten() # convert probabilities to binary predictions
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
    
    # workaround to extract true labels from the generator (since it's a generator, we can't directly access the labels with '.classes')
    y_true = np.concatenate([test_gen[i][1] for i in range(len(test_gen))])

    # map predicted and true labels to their original class names
    multiclass_labels = ['Adenosis', 'Ductal Carcinoma', 'Fibroadenoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma', 'Phyllodes Tumor', 'Tubular Adenoma']
    binary_labels = ['Benign', 'Malignant']
    
    # select the appropriate class names based on the classification type
    class_names = multiclass_labels if classification_type == 'multiclass' else binary_labels

    # map predicted and true labels to their original class names so that it is easier to interpret the results
    y_pred_labels = [class_names[i] for i in y_pred]
    y_true_labels = [class_names[i] for i in y_true]

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

    print(f"\nAccuracy: {accuracy:.4f}\n")
    
    # print classification report in a tabulated format (more readable and aesthetic)
    print("Classification Report:")
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = [
        [label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
        for label, metrics in report.items() if label not in ['accuracy', 'macro avg', 'weighted avg']
    ]
    print(tabulate(rows, headers, tablefmt='grid'))

    # overall metrics
    overall_headers = ["Metric", "Value"]
    overall_rows = [
        ["Accuracy", report['accuracy']],
        ["Macro Avg Precision", report['macro avg']['precision']],
        ["Macro Avg Recall", report['macro avg']['recall']],
        ["Macro Avg F1-Score", report['macro avg']['f1-score']],
        ["Weighted Avg Precision", report['weighted avg']['precision']],
        ["Weighted Avg Recall", report['weighted avg']['recall']],
        ["Weighted Avg F1-Score", report['weighted avg']['f1-score']],
    ]
    print("\nOverall Metrics:")
    print(tabulate(overall_rows, overall_headers, tablefmt='grid'))

    if show_confusion_matrix:
        print("\nConfusion Matrix:")
        plot_confusion_matrix(y_true_labels, y_pred_labels, class_names)
    else:
        plot_confusion_matrix(y_true_labels, y_pred_labels, class_names, save_to_file=True)
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }
    print("Evaluation complete. Returning results.")
    return results

def keras_tuning_model_builder(hp, input_shape=(224, 224, 3), classification_type='binary'):
    
    """
    Build a CNN model with a hyperparameter search space for tuning.

    This function constructs a Convolutional Neural Network (CNN) model adaptable for both binary and multiclass 
    image classification tasks. The model architecture includes convolutional layers, dense layers, and dropout layers. 
    Hyperparameters for filters, dense units, dropout rates, and learning rates are defined using the Keras Tuner.

    Parameters:
        - hp (HyperParameters): The Keras Tuner HyperParameters object for defining the search space.
        - input_shape (tuple, optional): The shape of the input images (height, width, channels). Defaults to (224, 224, 3).
        - num_classes (int, optional): The number of output classes. Use 2 for binary classification and >2 for multiclass classification. Defaults to 2.

    Returns:
        - model (Sequential): The compiled Keras model ready for hyperparameter tuning.
    """

    # ------------------------------
    # Initialize the Model
    # ------------------------------
    model = Sequential()

    # ------------------------------
    # Convolutional and Pooling Layers
    # ------------------------------

    conv_filter_1 = hp.Int('conv_filter_1', min_value=64, max_value=128, step=32)
    model.add(Conv2D(filters=conv_filter_1, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    conv_filter_2 = hp.Int('conv_filter_2', min_value=128, max_value=256, step=64)
    model.add(Conv2D(filters=conv_filter_2, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    conv_filter_3 = hp.Int('conv_filter_3', min_value=256, max_value=512, step=64)
    model.add(Conv2D(filters=conv_filter_3, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    conv_filter_4 = hp.Int('conv_filter_4', min_value=512, max_value=1024, step=128)
    model.add(Conv2D(filters=conv_filter_4, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    conv_filter_5 = hp.Int('conv_filter_5', min_value=512, max_value=1024, step=128)
    model.add(Conv2D(filters=conv_filter_5, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ------------------------------
    # Flatten Layer
    # ------------------------------
    model.add(Flatten())

    # ------------------------------
    # Dense and Dropout Layers
    # ------------------------------

    dense_units_0 = hp.Int('dense_units_0', min_value=256, max_value=512, step=64)
    model.add(Dense(units=dense_units_0, activation='relu'))

    dropout_rate_0 = hp.Float('dropout_rate_0', min_value=0, max_value=0.25, step=0.1)
    model.add(Dropout(rate=dropout_rate_0))

    # ------------------------------
    # Output Layer
    # ------------------------------
    if classification_type == 'binary':
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(8, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    # ------------------------------
    # Compile the Model
    # ------------------------------
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )

    return model