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

def plot_distribution_pie(data, col, figsize=(10, 5)):
    counts = data[col].value_counts()
    counts.plot(kind='pie', figsize=figsize, autopct='%1.1f%%', labels=[f'{label} ({count})' for label, count in zip(counts.index, counts)])
    plt.ylabel('')
    plt.title(f'Distribution of {col}')
    plt.show()

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