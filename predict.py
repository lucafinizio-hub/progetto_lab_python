import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================
# 1. Caricamento modello
# =========================
model = load_model("best_mnist_cnn.h5")

# =========================
# 2. Canvas per disegno
# =========================
canvas = np.ones((280, 280), dtype=np.uint8) * 255
drawing = False

def draw(event, x, y, flags, param):
    global drawing, canvas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 12, (0,), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw a digit")
cv2.setMouseCallback("Draw a digit", draw)

# =========================
# 3. Loop disegno
# =========================
while True:
    cv2.imshow("Draw a digit", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # predizione
        break
    elif key == ord('c'):  # pulisci
        canvas[:] = 255
    elif key == ord('e'):  # esci
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# =========================
# 4. Preprocessing avanzato
# =========================

# Resize
img = cv2.resize(canvas, (28, 28))

# Inversione colori (MNIST style)
img = 255 - img

# Blur per smoothing
img = cv2.GaussianBlur(img, (3, 3), 0)

# Threshold (pulizia)
_, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

# =========================
# 5. Centering (CRUCIALE)
# =========================
coords = np.column_stack(np.where(img > 0))

if len(coords) > 0:
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img_cropped = img[y_min:y_max, x_min:x_max]

    h, w = img_cropped.shape
    scale = 20.0 / max(h, w)

    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_cropped, (new_w, new_h))

    img_final = np.zeros((28, 28))

    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    img_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    img = img_final

# =========================
# 6. Normalizzazione
# =========================
img = img / 255.0

# =========================
# 7. Reshape per CNN
# =========================
img = img.reshape(1, 28, 28, 1)

# =========================
# 8. Predizione
# =========================
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction)

print(f"Predizione: {predicted_digit} (confidenza: {confidence:.4f})")

# =========================
# 9. Visualizzazione
# =========================
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_digit} ({confidence:.2f})")
plt.axis('off')
plt.show()