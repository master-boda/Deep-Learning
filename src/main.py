from utils.preproc import preproc_pipeline
from utils.modeling import *
from utils.model_instances import *
from utils.visualizations import *

import tensorflow as tf
import json
from tqdm import tqdm

# for releasing the GPU memory (only useful if using CUDA)
from numba import cuda
import numpy as np

# CUDA (Nvidia GPU Computing)
if len(tf.config.list_physical_devices('GPU')) > 0:
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    
    device = cuda.get_current_device()
    device.reset() # dump the memory contents to free up the memory (it accumulates over the session)
    
    tf.config.experimental.set_memory_growth(gpus[0], True)

    tf.config.set_logical_device_configuration(
    gpus[0], 
    [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])  # limit to 4GB

    tf.compat.v1.disable_eager_execution()
    
def main(model,
         classification_type,
         model_name,
         save_model_instance=True,
         epochs=40,
         early_stopping_patience=10,
         lr_reduction_patience=3,
         lr_reduction_factor=0.1,
         min_lr=1e-7,
         batch_size=32,
         verbose=False,
         data_augmentation=True,
         augmented_images_per_image=5,
         image_resolution=(224, 224)):
    """
    Main function to train and evaluate a model.
    
    Parameters:
        - model (tf.keras.Model): The model to be trained.
        - classification_type (str): The type of classification ('binary' or 'multiclass').
        - model_name (str): The name of the model to be saved.
        - save_model_instance (bool): Whether to save the trained model instance.
        - data_augmentation (bool): Whether to perform data augmentation on the training dataset.
        - augmented_images_per_image (int): Number of augmented images to generate per original image.
        - image_resolution (tuple): The desired resolution to resize the images (width, height).
        - epochs (int): The number of epochs to train the model.
        - early_stopping_patience (int): The number of epochs with no improvement after which training will be stopped.
        - lr_reduction_patience (int): The number of epochs with no improvement after which the learning rate will be reduced.
        - lr_reduction_factor (float): The factor by which the learning rate will be reduced.
        - min_lr (float): The minimum learning rate.
        - batch_size (int): The batch size for the data generators.
        - verbose (bool): Whether to print verbose output.
        
    Returns:
        - model (tf.keras.Model): The trained model.
    """
    
    train_gen, val_gen, test_gen, class_weights, steps_per_epoch = preproc_pipeline(image_resolution=image_resolution, 
                                                                                    classification_type=classification_type,
                                                                                    use_data_augmentation=data_augmentation,
                                                                                    augmented_images_per_image=augmented_images_per_image,
                                                                                    batch_size=batch_size,
                                                                                    verbose=verbose)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_reduction_factor, patience=lr_reduction_patience, min_lr=min_lr)
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
    callbacks = [reduce_lr, early_stopping]
    
    trained_model = train_model(train_gen, val_gen, model, callbacks=callbacks, epochs=epochs, class_weights=class_weights, steps_per_epoch=steps_per_epoch)
    
    evaluate_model(trained_model, classification_type=classification_type)
    
    if save_model_instance:
        save_model(trained_model, f'src/models/{model_name}.h5')
    
    return trained_model

def main_loop(model_name, classification_type='multiclass', data_augmentation=True, augmented_images_per_image=5):
    
    results = []

    for trainable_layers in tqdm(range(4, 16), desc="Trainable Layers"): # progress bar using tqdm
        print(f"Training with {trainable_layers} trainable layers...")
        model = multiclass_classification_vgg16_model(trainable_layers=trainable_layers)
        
        trained_model = main(model, 
                             classification_type=classification_type, 
                             model_name=model_name, 
                             save_model_instance=False,
                             epochs=40, 
                             early_stopping_patience=10, 
                             batch_size=32, 
                             verbose=False, 
                             data_augmentation=data_augmentation, 
                             augmented_images_per_image=augmented_images_per_image, 
                             image_resolution=(224, 224))
        
        result = evaluate_model(trained_model, classification_type=classification_type)
        results.append({
            'model_architecture': model_name + ' (' + classification_type + ')',
            'trainable_layers': trainable_layers,
            'results': result
        })
        
        print(f"Completed training with {trainable_layers} trainable layers.\n")
        print(f"Results: {result}")


    with open('src/models/result_logs/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to src/models/result_logs/results.json")


model = multiclass_classification_vgg16_model(learning_rate=1e-4, trainable_layers=7)

main(model=model, classification_type='multiclass', model_name='VGG16_multiclass')