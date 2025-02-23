import numpy as np
from src.mobile.coreml_inference import coreml_inference

def test_coreml_inference():
    model_path = "models/model.mlmodel"
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    predictions = coreml_inference(model_path, input_data)
    assert predictions is not None

if __name__ == "__main__":
    test_coreml_inference()