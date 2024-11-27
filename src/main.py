# PREPROC PIPELINE
# MODELING PIPELINE

#from src.preproc import preproc_pipeline

import tensorflow as tf
print("Num CPUs Available: ", tf.config.list_physical_devices('GPU'))

import os
print("CWD:", os.getcwd())

# check if src module exists
print("src module exists:", os.path.exists("src"))

#X_train, y_train, X_test, y_test, X_val, y_val = preproc_pipeline(desired_magnification='40X', 
#                                                    image_resolution=(224, 224), 
#                                                    classification_type='binary')