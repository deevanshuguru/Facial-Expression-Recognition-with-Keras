# To detect emotion in stored picture
import cv2
import numpy as np
from keras.models import load_model
import sys


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

conf = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=conf))


frontalface = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
model = load_model('models/kaggle_model.hdf5')

def test_image(addr):
    target = ['angry','disgust','fear','happy','sad','surprise','neutral']
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    im = cv2.imread(addr) 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = frontalface.detectMultiScale(gray,scaleFactor=1.1)
    
    for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2,5)
            face = im[y:y+h,x:x+w]
            face = cv2.resize(face,(48,48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = face.astype('float32')/255
            face = np.asarray(face)
            face = face.reshape(1, 1,face.shape[0],face.shape[1])
            result = target[np.argmax(model.predict(face))]
            cv2.putText(im,result,(x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)
            
    cv2.imshow('result', im)
    cv2.imwrite('result_image.jpg',im) ''' '''
    cv2.waitKey(0) 
    
if __name__=='__main__':
    image_addres = sys.argv[1]
    test_image(image_addres)