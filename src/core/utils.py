import numpy as np

def load_data(data_path):
    # Load sample data (e.g., .tflite or .npy)
    return np.load(data_path)

def preprocess_input(data):
    # Normalize and resize input data
    data = data / 255.0  # Normalize to [0, 1]
    data = np.expand_dims(data, axis=0)  # Add batch dimension
    return data.astype(np.float32)