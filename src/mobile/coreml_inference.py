import coremltools as ct

def coreml_inference(model_path, input_data):
    model = ct.models.MLModel(model_path)
    predictions = model.predict({"input": input_data})
    return predictions