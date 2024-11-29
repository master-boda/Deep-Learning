from utils.preproc import preproc_pipeline
from utils.modeling import *
from utils.visualizations import *

from models.models import *

import tensorflow as tf

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

def main(model, classification_type, image_resolution, verbose=True):

    magnifications =  {'40X' : {'accuracy': 0},
                      '100X' : {'accuracy': 0},
                      '200X' : {'accuracy': 0},
                      '400X' : {'accuracy': 0}}
    
    for magnification in magnifications:
        train_gen, val_gen, X_test, y_test, class_weights = preproc_pipeline(desired_magnification=magnification, 
                                                    image_resolution=image_resolution, 
                                                    classification_type=classification_type)
        
        model_instance = model(input_shape=(image_resolution[0], image_resolution[1], 3))
        
        fitted_model = train_model(train_gen, val_gen, model_instance, class_weights=class_weights, epochs=8)
        
        test_loss, test_acc = fitted_model.evaluate(X_test, y_test)
        
        if verbose:
            print(f'Model: {model}, Magnification: {magnification}')
            print(f'Test loss: ', test_loss)
            print(f'Test accuracy: ', test_acc)
        
        magnifications[magnification]['accuracy'] = test_acc
        
    return magnifications

print(main(binary_classification_baseline_model, 'binary', (224, 224)))

# 1st Try Results (accuracy, 8 epochs):
# binary_classification_baseline_model: {'40X': 0.73333335, '100X': 0.71794873, '200X': 0.84437084, '400X': 0.86080587}
# multiclass_classification_baseline_model : {'40X': 0.10666667, '100X': 0.3621795, '200X': 0.2615894, '400X': 0.35897437}