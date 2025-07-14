import tensorflow as tf

# Load your trained Keras model
model = tf.keras.models.load_model('recycle_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('recycle_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Converted to TFLite and saved as recycle_model.tflite")
