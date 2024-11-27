from preproc import *

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

# If CUDA is available it will print a non-empty list
print("Num CPUs Available: ", tf.config.list_physical_devices('GPU'))

X_train, y_train, X_test, y_test, X_val, y_val = preproc_pipeline(desired_magnification='40X', 
                                                    image_resolution=(224, 224), 
                                                    classification_type='binary')