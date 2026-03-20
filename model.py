import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib

# Caricamento modello
model = joblib.load("mnist_model.pkl")

# Canvas
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

while True:
    cv2.imshow("Draw a digit", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # predizione
        break
    elif key == ord('c'):  # pulizia
        canvas[:] = 255
    elif key == ord('e'):  # uscita
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# Preprocessing
img = cv2.resize(canvas, (28, 28))
img = 255 - img
img = cv2.GaussianBlur(img, (3, 3), 0)
img = img / 255.0
img_flat = img.reshape(1, -1)

# Predizione
prediction = model.predict(img_flat)

print("Predizione:", prediction[0])

# Visualizzazione
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
plt.axis('off')
plt.show()