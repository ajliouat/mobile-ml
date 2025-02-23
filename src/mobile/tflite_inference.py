from src.core.inference import run_inference

def tflite_inference(model_path, input_data):
    return run_inference(model_path, input_data)