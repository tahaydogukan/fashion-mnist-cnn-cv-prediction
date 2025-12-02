import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1) Kendi modelimizi yüklüyoruz
model = load_model("my_model.keras")

# 2) Modelimizin tahmin edeceği sınıf isimlerini yüklüyoruz
class_names = [
    "T-shirt/Top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot"    # 9
]

# 3) Kamerayı açıyoryuz
cap = cv2.VideoCapture(0) # 0-> Default Kamerayı temsil ediyor.

while True:
    ref, frame = cap.read()
    if not ref:
        break
    
    # 4) Görüntüyü aynalıyoruz
    frame = cv2.flip(frame,1)

    # 5) Görüntüyü Gri Tonlara Çeviriyoruz
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 6) Görüntüyü 28x28 formota getiriypruz ve normalize ediyoruz.
    img = cv2.resize(gray,(28,28)) / 255

    # 7) CNN' e uygun formata getiriyoruz veriyi
    input_img = img.reshape(1,28,28,1)

    # 8) Modelimiz ile tahmin yapıyoruz
    preds = model.predict(input_img,verbose=0)
    pred_class= np.argmax(preds)
    confidence = np.max(preds)

    label = f"{class_names[pred_class]} (%{confidence*100:.1f})"

    # 9) Sonuçları ekranda gösteriyoruz
    cv2.putText(frame,label,(20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Fashion-Mnist", frame)

    # Q'ya basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
