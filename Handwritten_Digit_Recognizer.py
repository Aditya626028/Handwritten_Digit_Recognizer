# Handwritten Digit Recognizer using CNN (MNIST Dataset)

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. Load Dataset (MNIST)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2. Normalize data (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Reshape (add channel for CNN: 28x28x1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# 4. Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 5. Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train Model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 7. Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test accuracy: {test_acc*100:.2f}%")

# 8. Save Model
model.save("digit_recognizer.h5")
print("Model saved as digit_recognizer.h5")

# 9. Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()
