def switch_model_based_on_battery(battery_level, battery_threshold, model_path):
    if battery_level < battery_threshold:
        return "models/model_quantized.tflite"  # Switch to a lightweight model
    return model_path  # Use the default model