import numpy as np
from src.mobile.tflite_inference import tflite_inference

def test_tflite_inference():
    model_path = "models/model.tflite"
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    predictions = tflite_inference(model_path, input_data)
    assert predictions is not None

if __name__ == "__main__":
    test_tflite_inference()