import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Caricamento dataset MNIST
mnist = fetch_openml('mnist_784', version=1)

# Conversione in numpy
X = mnist.data.to_numpy()
y = mnist.target.to_numpy()

# Normalizzazione
X = X / 255.0

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modello
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=30,
    solver='adam',
    random_state=42,
    verbose=True
)

# Training
model.fit(X_train, y_train)

# Valutazione
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train accuracy:", accuracy_score(y_train, y_pred_train))
print("Test accuracy:", accuracy_score(y_test, y_pred_test))

# Salvataggio modello
joblib.dump(model, "mnist_model.pkl")

print("Modello salvato come mnist_model.pkl")