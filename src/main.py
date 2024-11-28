from utils.preproc import preproc_pipeline
from utils.modeling import *
from utils.visualizations import *

from models.models import *

import tensorflow as tf

# For releasing the GPU memory (only useful if using CUDA)
from numba import cuda

# CUDA (Nvidia GPU Computing)
if len(tf.config.list_physical_devices('GPU')) > 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    
    device = cuda.get_current_device()
    device.reset() # Dump the memory contents to free up the memory (it accumulates over the session)
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) # Let Tensorflow allocate memory as needed

X_train, y_train, X_test, y_test, X_val, y_val = preproc_pipeline(desired_magnification='200X', 
                                                    image_resolution=(224, 224), 
                                                    classification_type='binary')

model = binary_classification_baseline_model(input_shape=(224, 224, 3))

fitted_model = train_model(X_train, y_train, X_val, y_val, model, epochs=8)

test_loss, test_acc = fitted_model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
