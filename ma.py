from cv2 import cv2
from keras.models import load_model
import numpy as np
maxterm=0
model=load_model('my_model.h5')
res_dict={0:"angry",1:"disgusted",2:"Fearful",3:"happy",4:"neutral",5:"sad",6:"surprised",}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cropped_img=np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        emotion_prediction = model.predict(cropped_img)
        maxterm = int(np.argmax(emotion_prediction))


    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, res_dict[maxterm], (15, 25), font, 1, color=(0, 225, 0), thickness=1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()