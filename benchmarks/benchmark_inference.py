import time
from src.core.inference import run_inference
import numpy as np

def benchmark_inference(model_path, input_data, num_runs=100):
    # Warm-up run
    run_inference(model_path, input_data)

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        run_inference(model_path, input_data)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
    print(f"Average inference time: {avg_time:.2f} ms")

if __name__ == "__main__":
    model_path = "models/model.tflite"
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Example input
    benchmark_inference(model_path, input_data)