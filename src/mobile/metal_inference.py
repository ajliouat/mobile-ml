import coremltools as ct

def metal_inference(model_path, input_data):
    model = ct.models.MLModel(model_path)
    model.compute_units = ct.ComputeUnit.CPU_AND_GPU  # Use Metal acceleration
    predictions = model.predict({"input": input_data})
    return predictions