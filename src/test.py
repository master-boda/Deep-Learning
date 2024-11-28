from utils.preproc import *
from utils.modeling import *
from models.models import *

import tensorflow as tf

# If CUDA is available it will print a non-empty list
print("Num CPUs Available: ", tf.config.list_physical_devices('GPU'))

X_train, y_train, X_test, y_test, X_val, y_val = preproc_pipeline(desired_magnification='200X', 
                                                    image_resolution=(224, 224), 
                                                    classification_type='binary')

print('X_train type:', type(X_train),
    'X_val type:', type(X_val),
    'X_test type:', type(X_test),
    'y_train type:', type(y_train),
    'y_val type:', type(y_val),
    'y_test type:', type(y_test))

model = binary_classification_baseline_model(input_shape=(224, 224, 3))

fitted_model = train_model(X_train, y_train, X_val, y_val, model, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
