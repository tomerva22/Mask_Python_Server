from flask import Flask, jsonify, request
from flask.helpers import make_response
from flask.wrappers import Response
from flask_cors import CORS
import cv2
import os
import magic
import numpy as np
import tensorflow as tf

def detect_mask(result_file):
    model = tf.keras.models.load_model("mobileNetV2FaceMask.model")
    labels_dict={0:'without_mask',1:'with_mask'}
    color_dict={0:(0,0,255),1:(0,255,0)}
    size=4
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(result_file)

    #if mime_type == 'image/jpeg':
    #    os.rename(result_file, result_file + '.jpg')
    #    result_file = result_file + '.jpg'
    #elif mime_type == 'image/png':
    #    os.rename(result_file, result_file + '.png')
    #    result_file = result_file + '.png'
    #elif mime_type == 'image/gif':
    #    os.rename(result_file, result_file + '.gif')
    #    result_file = result_file + '.gif'
    #elif mime_type == 'image/bmp':
    #    os.rename(result_file, result_file + '.bmp')
    #    result_file = result_file + '.bmp'
    #elif mime_type == 'image/tiff':
    #    os.rename(result_file, result_file + '.tiff')
    #    result_file = result_file + '.tiff'
    #else:
    #    print('Not an image? %s' % mime_type)

    im=cv2.imread(result_file)
    im=cv2.flip(im,1,1) #Flip to act as a mirror
    
    # Resize the image to speed up detection
    mini =  im.reshape(-1, im.shape[0], im.shape[1], im.shape[2]) #cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    #print(mini.shape)
    #print(mini.dim)
    # detect MultiScale / faces 
    #mini,1.05,5
    faces = classifier.detectMultiScale(mini,1.05,5)# you may change params or 
    #remove them if faces not detected decrease them if u get falses then increase them

    normalized=mini/255.0
    result=model.predict(normalized)
    label = np.argmax(result,axis=1)[0]
    print(labels_dict[label])
    response = make_response(labels_dict[label])
    print(response)
    return response
        #cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        #cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        #cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)



def detect_face(path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread(path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.05, 3)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    detect_face('C:\\Users\\tomer\\Downloads\\front-slide-6.jpg')
    detect_face('F:\\MLBig\\CNN\\1\\data\\test\\p\\85.jpg')
    detect_mask('F:\\MLBig\\CNN\\1\\data\\test\\p\\86.jpg')
    #detect_mask('F:\\MLBig\\CNN\\1\\data\\n\\5.jpg')
    #detect_mask('F:\\MLBig\\CNN\\1\\data\\n\\6.jpg')
    #detect_mask('F:\\MLBig\\CNN\\1\\data\\n\\11.jpg')
    #detect_mask('F:\\MLBig\\CNN\\1\\data\\n\\14.jpg')
    #detect_mask('F:\\MLBig\\CNN\\1\\data\\n\\17.jpg')
    #detect_mask('F:\\MLBig\\CNN\\1\\data\\n\\8.jpg')
    #detect_mask('F:\\MLBig\\CNN\\1\\data\\n\\9.jpg')