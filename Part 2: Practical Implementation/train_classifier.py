import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1️⃣ Generate fake data
X_train = np.random.rand(80, 32, 32, 3)
y_train = np.random.randint(0, 2, size=(80,))
X_test  = np.random.rand(20, 32, 32, 3)
y_test  = np.random.randint(0, 2, size=(20,))

# 2️⃣ Build a simple CNN
model = models.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 3️⃣ Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4️⃣ Train
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 5️⃣ Plot
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.show()

# 6️⃣ Save
model.save('recycle_model.h5')
print("✅ Model saved as recycle_model.h5")

