import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Caricamento Dati
mnist = fetch_openml('mnist_784')



# Conversione in Numpy
X = mnist.data.to_numpy()
y = mnist.target.to_numpy()



# Normalizzazione
X = X / 255.0



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Modello
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=20,
    solver='adam',
    random_state=42,
    verbose=True
)

model.fit(X_train, y_train)




# Valutazione
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train accuracy:", accuracy_score(y_train, y_pred_train))
print("Test accuracy:", accuracy_score(y_test, y_pred_test))




# Foglio di disegno
canvas = np.ones((280, 280), dtype=np.uint8) * 255
drawing = False

def draw(event, x, y, flags, param):
    global drawing, canvas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 8, (0,), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw a digit")
cv2.setMouseCallback("Draw a digit", draw)

print("Disegna una cifra (0-9), premi 'q' per predire")

while True:
    cv2.imshow("Draw a digit", canvas)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('c'):  # pulisci canvas
        canvas[:] = 255

cv2.destroyAllWindows()




#Preprocessing immagine

# Ridimensiona a 28x28
img = cv2.resize(canvas, (28, 28))

# Inverti colori (MNIST = bianco su nero)
img = 255 - img

# Blur leggero (migliora risultato)
img = cv2.GaussianBlur(img, (3, 3), 0)

# Normalizza
img = img / 255.0

# Flatten
img_flat = img.reshape(1, -1)





# Predizione
prediction = model.predict(img_flat)



# Visualizzazione
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
plt.axis('off')
plt.show()