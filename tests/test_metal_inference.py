import numpy as np
from src.mobile.metal_inference import metal_inference

def test_metal_inference():
    model_path = "models/model.mlmodel"
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    predictions = metal_inference(model_path, input_data)
    assert predictions is not None

if __name__ == "__main__":
    test_metal_inference()