import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("models/cat_dog_classifier.h5")

labels = ["Cat 🐱", "Dog 🐶"]

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

  
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

   
    pred = model.predict(img, verbose=0)[0][0]

    if pred > 0.5:
        label = labels[1]
    else:
        label = labels[0]

   
    cv2.putText(
        frame,
        label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Cat vs Dog Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
