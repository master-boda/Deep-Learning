import numpy as np
import tensorflow as tf
import os

from src.utils.preproc import *

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from keras.callbacks import EarlyStopping

from src.utils.visualizations import plot_confusion_matrix

from tabulate import tabulate

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

def save_model(model, path="src/models", model_name="saved_model.h5"):
    """
    Save the TensorFlow model to the specified path in HDFS5 format.

    Parameters:
    - model: TensorFlow model to be saved.
    - path: Destination path where the model will be saved. Default is "src/models".
    - model_name: Name of the model file. Default is "saved_model.h5".
    
    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(os.path.join(path, model_name), save_format='h5')

def evaluate_model(model, classification_type='binary', desired_magnification='40X'):
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
    train_gen, val_gen, test_gen, class_weights = preproc_pipeline(desired_magnification, (224, 224), classification_type=classification_type, use_data_augmentation=False)
    
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

    plot_confusion_matrix(y_true_labels, y_pred_labels, class_names)
    print("Plotted confusion matrix.")
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }
    print("Evaluation complete. Returning results.")
    return results