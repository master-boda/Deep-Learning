import csv
from datetime import datetime
import os

def log_results_to_csv(file_path, model_name, dataset_name, hyperparameters, metrics):
    # Example usage
    #log_results_to_csv('model_results.csv', 'ResNet50', 'Dataset1', 'learning_rate=0.001', 'accuracy=0.95, loss=0.05')
    
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file does not exist
        if not file_exists:
            writer.writerow(['Timestamp', 'Model', 'Dataset', 'Hyperparameters', 'Metrics'])
        
        # Write the log entry
        writer.writerow([datetime.now(), model_name, dataset_name, hyperparameters, metrics])
