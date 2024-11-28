import pandas as pd
import matplotlib.pyplot as plt 
import cv2
from PIL import Image

def check_image_resolutions(metadata):
    resolutions = []
    for idx, row in metadata.iterrows():
        try:
            with Image.open(row['path_to_image']) as img:
                resolutions.append(img.size)
        except Exception as e:
            print(f"Error loading image {row['path_to_image']}: {e}")
    return resolutions

def plot_distribution_pie(data, col, ax=None, figsize=(10, 5)):
    counts = data[col].value_counts()
    counts.plot(kind='pie', autopct='%1.1f%%', labels=[f'{label} ({count})' for label, count in zip(counts.index, counts)], ax=ax)
    if ax is None:
        plt.ylabel('')
        plt.title(f'Distribution of {col}')
        plt.show()
    else:
        ax.set_ylabel('')
        ax.set_title(f'Distribution of {col}')

def plot_images_compare_magnification(data, cancer_types, magnifications):
    fig, axes = plt.subplots(figsize=(25, 30), sharey=True, ncols=4, nrows=8)

    for j, cancer_type in enumerate(cancer_types):
        for i, mag in enumerate(magnifications):
            temp_df = data[(data['Cancer Type'] == cancer_type) & (data['Magnification'] == mag)]

            if temp_df.empty:
                continue

            random_id = temp_df.sample(n=1).index[0]
            image = cv2.imread(data['path_to_image'][random_id])

            axes[j][i].imshow(image)
            axes[j][i].set_title(f"{cancer_type} (Mag.= {mag})")
            axes[j][i].axis('off')
            
def plot_training_history(model):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axes[0].plot(model.history['loss'], label='train')
    axes[0].plot(model.history['val_loss'], label='val')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(model.history['accuracy'], label='train')
    axes[1].plot(model.history['val_accuracy'], label='val')
    axes[1].set_title('Accuracy')
    axes[1].legend()

    plt.show()