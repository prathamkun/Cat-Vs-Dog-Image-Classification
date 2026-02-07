import cv2
import numpy as np
import tensorflow as tf



model = tf.keras.models.load_model("models/cat_dog_classifier.h5")
   

IMG_SIZE = 224
CLASS_NAMES = ["Cat", "Dog"]   



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

print("Camera started... press Q to quit")



while True:

    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.flip(frame, 1)

   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

  
    preds = model.predict(img, verbose=0)[0]

    class_index = np.argmax(preds)
    confidence = preds[class_index]

    label = CLASS_NAMES[class_index]

    

    h, w, _ = frame.shape

    
    box_size = 300
    x1 = int(w/2 - box_size/2)
    y1 = int(h/2 - box_size/2)
    x2 = int(w/2 + box_size/2)
    y2 = int(h/2 + box_size/2)

    
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 3)

    
    text = f"{label} : {confidence:.2f}"

    cv2.putText(frame,
                text,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),   
                2)


    cv2.imshow("CatDog AI", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
