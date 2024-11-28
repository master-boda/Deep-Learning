from utils.preproc import preproc_pipeline
from utils.modeling import *
from utils.visualizations import *

from models.models import *

import tensorflow as tf

# For releasing the GPU memory (only useful if using CUDA)
from numba import cuda
import numpy as np

# CUDA (Nvidia GPU Computing)
if len(tf.config.list_physical_devices('GPU')) > 0:
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    
    device = cuda.get_current_device()
    device.reset() # Dump the memory contents to free up the memory (it accumulates over the session)
    
    tf.config.experimental.set_memory_growth(gpus[0], True)

    tf.config.set_logical_device_configuration(
    gpus[0], 
    [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])  # Limit to 4GB

    tf.compat.v1.disable_eager_execution()


X_train, y_train, X_test, y_test, X_val, y_val = preproc_pipeline(desired_magnification='200X', 
                                                    image_resolution=(224, 224), 
                                                    classification_type='binary')

model = binary_classification_baseline_model(input_shape=(224, 224, 3))

fitted_model = train_model(X_train, y_train, X_val, y_val, model, epochs=8, data_gen=True)

test_loss, test_acc = fitted_model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)