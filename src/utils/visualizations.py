import pandas as pd
import matplotlib.pyplot as plt 
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import random

def check_image_resolutions(metadata):
    """
    Check the resolutions of images listed in the metadata.
    
    Parameters:
        - metadata (pandas.DataFrame): A DataFrame containing image metadata (containing 'path_to_image' column).
        
    Returns:
        - resolutions (list): A list of tuples where each tuple contains the width and height of an image.
    """
    resolutions = []
    for idx, row in metadata.iterrows():
        with Image.open(row['path_to_image']) as img:
            resolutions.append(img.size)

    return resolutions

def plot_distribution_pie(data, col, ax=None, figsize=(10, 5)):
    """
    Plots a pie chart showing the distribution of values in a specified column of a DataFrame.
    Capable of displaying multiple pie charts in the same figure (useful for side by side comparisons).
    
    Parameters:
        - data (pd.DataFrame): The DataFrame containing the data.
        - col (str): The column name to plot the distribution.
        - ax (matplotlib.axes.Axes, optional): The axes on which to plot the pie chart. If None, a new figure and axes are created. Default is None.
        - figsize (tuple, optional): The size of the figure if ax is None. Default is (10, 5).
        
    Returns:
        None
    """
    counts = data[col].value_counts()
    # autopct='%1.1f%%' formats the percentage to have one decimal place
    # zip(counts.index, counts) contains the labels for the selected "col" and the respective counts
    counts.plot(kind='pie', autopct='%1.1f%%', labels=[f'{label} ({count})' for label, count in zip(counts.index, counts)], ax=ax)
    
    # ax used for displaying multiple plots in the same figure (useful for comparing distributions)
    if ax is None:
        plt.ylabel('')
        plt.title(f'Distribution of {col}')
        plt.show()
    else:
        ax.set_ylabel('')
        ax.set_title(f'Distribution of {col}')

def plot_images_compare_magnification(data, cancer_types, magnifications):
    """
    Plots a comparison of images from different magnifications for each cancer type.
    
    Parameters:
        - data (pd.DataFrame): DataFrame containing the image data with the columns 'Cancer Type', 'Magnification', and 'path_to_image'.
        - cancer_types (list): List of cancer types to be compared.
        - magnifications (list): List of magnifications to be compared.
        
    Returns:
        None
    """
    # function to compare images from different magnifications for each cancer type
    fig, axes = plt.subplots(figsize=(25, 30), sharey=True, ncols=4, nrows=8)

    for j, cancer_type in enumerate(cancer_types):
        for i, mag in enumerate(magnifications):
            # filter the data for the selected cancer type and magnification
            temp_df = data[(data['Cancer Type'] == cancer_type) & (data['Magnification'] == mag)]

            # select a random
            random_id = temp_df.sample(n=1).index[0]
            image = cv2.imread(data['path_to_image'][random_id])

            axes[j][i].imshow(image)
            axes[j][i].set_title(f"{cancer_type} (Mag.= {mag})")
            axes[j][i].axis('off')
            
def plot_confusion_matrix(y_true, y_pred, class_names, save_to_file=False, filename='confusion_matrix.png'):
    """
    Plot confusion matrix using seaborn heatmap.
    
    Parameters:
        - y_true (array-like): The true labels.
        - y_pred (array-like): The predicted labels.
        - class_names (list): The names of the classes.
        
    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_to_file:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
            
def plot_training_history(model, metric='loss', title='Model Loss'):
    """
    Plots the training history of a TensorFlow model.
    
    Parameters:
        - model (tf.keras.Model): The trained TensorFlow model.
        - metric (str, optional): The metric to plot. Default is 'loss'.
        - title (str, optional): The title of the plot. Default is 'Model Loss'.
        
    Returns:
        None
    """
    plt.plot(model.history.history[metric])
    plt.plot(model.history.history[f'val_{metric}'])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
def plot_images_cancer_types(data, benign_types, malignant_types, magnifications):
    """
    Plots a comparison of images from different magnifications for each cancer type.
    
    Parameters:
        - data (pd.DataFrame): DataFrame containing the image data with the columns 'Cancer Type', 'Magnification', and 'path_to_image'.
        - benign_types (list): List of benign cancer types to be compared.
        - malignant_types (list): List of malignant cancer types to be compared.
        - magnifications (list): List of magnifications to be compared.
        
    Returns:
        None
    """
    nrows = 2  # two rows: one for benign, one for malignant
    ncols = 4  # four columns for each row
    
    fig, axes = plt.subplots(figsize=(25, 10), nrows=nrows, ncols=ncols, sharey=True)

    for j, cancer_type in enumerate(benign_types + malignant_types):
        row = 0 if cancer_type in benign_types else 1
        col = benign_types.index(cancer_type) if cancer_type in benign_types else malignant_types.index(cancer_type)

        for mag in magnifications:
            temp_df = data[(data['Cancer Type'] == cancer_type) & (data['Magnification'] == mag)]

            # random image
            random_id = temp_df.sample(n=1).index[0]
            image = cv2.imread(data.loc[random_id, 'path_to_image'])

            axes[row][col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[row][col].set_title(f"{cancer_type} (Mag.= {mag})", fontsize=18, fontweight='bold')
            axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()
    
def visualize_original_and_augmented_images(X, y, datagen, num_samples=5):
    """
    Visualizes original and augmented images side by side.

    Parameters:
        - X (numpy.ndarray): Array of images.
        - y (numpy.ndarray): Array of labels.
        - datagen (ImageDataGenerator): Keras ImageDataGenerator instance for augmenting images.
        - num_samples (int, optional): Number of samples to visualize. Defaults to 5.

    Returns:
        None
    """
    # mapping of the labels to their original class names (label encoder uses alphabetical order)
    label_mapping = {
        0: 'Adenosis',
        1: 'Ductal Carcinoma',
        2: 'Fibroadenoma',
        3: 'Lobular Carcinoma',
        4: 'Mucinous Carcinoma',
        5: 'Papillary Carcinoma',
        6: 'Phyllodes Tumor',
        7: 'Tubular Adenoma'
    }
    
    for i in random.sample(range(len(X)), num_samples):
        original_image = X[i]
        original_label = label_mapping[y[i]]
        # expand the dimensions to match the shape required by the generator (4D array)
        augmented_images = [datagen.flow(np.expand_dims(original_image, axis=0), batch_size=1).next()[0] for boda in range(5)]

        fig, axes = plt.subplots(1, 6, figsize=(20, 4))
        fig.suptitle(f'Original and Augmented Images - Label: {original_label}')

        axes[0].imshow(original_image)
        axes[0].set_title('Original')
        axes[0].axis('off')

        for j, aug_img in enumerate(augmented_images):
            axes[j + 1].imshow(aug_img)
            axes[j + 1].set_title(f'Augmented {j + 1}')
            axes[j + 1].axis('off')

        plt.show()