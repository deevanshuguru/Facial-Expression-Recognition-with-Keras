# To detect emotions in live media
import cv2
import numpy as np
import  as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.1
# CUDA_VISIBLE_DEVICES=1 python task.py
# export CUDA_VISIBLE_DEVICES=1
set_session(tf.Session(config=conf))

frontalface = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
''' Imported haarcascade_frontalface_alt2.xml '''

live_video = cv2.VideoCapture(0)
model = load_model('models/kaggle_model.hdf5')
model.get_config()

target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    # Capture frame-by-frame
    ret, frame = live_video.read()
    # Color conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detection
    faces = frontalface.detectMultiScale(gray, scaleFactor=1.1)

    # This will draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype('float32') / 255
        face = np.asarray(face)
        face = face.reshape(1, 1, face.shape[0], face.shape[1])
        result = target[np.argmax(model.predict(face))]
        cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done,this will release the captured one
live_video.release()
cv2.destroyAllWindows()
