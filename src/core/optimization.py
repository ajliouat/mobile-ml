import tensorflow as tf

def quantize_model(model_path, output_path, quantization_type="int8"):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    if quantization_type == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset  # Define this function
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    elif quantization_type == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

def representative_dataset():
    # Example: Provide a representative dataset for quantization
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]