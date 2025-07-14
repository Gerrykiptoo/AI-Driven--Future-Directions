import numpy as np
from PIL import Image
import sys
from keras.models import load_model

# Load the trained model
model = load_model('recycle_model.h5')

# Load and preprocess the image
img_path = sys.argv[1]
img = Image.open(img_path).resize((32, 32))  # Must match training size
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 32, 32, 3)   # add batch dimension

# Predict
prediction = model.predict(img_array)
print("Prediction (probability of recyclable):", prediction[0][0])
print("Predicted class:", "recyclable" if prediction[0][0] > 0.5 else "non-recyclable")
