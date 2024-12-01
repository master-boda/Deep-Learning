import pandas as pd
import matplotlib.pyplot as plt 
import cv2
from PIL import Image

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