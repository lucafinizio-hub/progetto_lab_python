import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# 1. Caricamento dataset
# =========================
mnist = fetch_openml('mnist_784', version=1)

X = mnist.data.to_numpy()
y = mnist.target.to_numpy().astype(int)

# =========================
# 2. Preprocessing
# =========================
# Normalizzazione
X = X / 255.0

# Reshape per CNN (28x28x1)
X = X.reshape(-1, 28, 28, 1)

# One-hot encoding
y = to_categorical(y, 10)

# =========================
# 3. Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Modello CNN
# =========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

# =========================
# 5. Compilazione
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 6. Callback
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/best_mnist_cnn.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# =========================
# 7. Training
# =========================
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# =========================
# 8. Valutazione
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"\nTest accuracy: {test_acc:.4f}")

# =========================
# 9. Salvataggio finale
# =========================
model.save("models/mnist_cnn_final.h5")