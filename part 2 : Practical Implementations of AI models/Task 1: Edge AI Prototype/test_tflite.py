import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="recycle_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare a dummy image (or you can load a real one, resized to 32x32)
sample = np.random.rand(1, 32, 32, 3).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], sample)
interpreter.invoke()

# Get the prediction
output = interpreter.get_tensor(output_details[0]['index'])
print("TFLite Prediction:", output)
