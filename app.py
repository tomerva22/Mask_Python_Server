from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import os
import magic
import numpy as np
import tensorflow as tf

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/mask', methods=['POST'])
def detect_mask():
    data = request.data
    model = tf.keras.models.load_model("mobileNetV2FaceMask.model")
    labels_dict={0:'without_mask',1:'with_mask'}
    color_dict={0:(0,0,255),1:(0,255,0)}
    size=4
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    result_file = "newfile"
    with open(result_file, 'wb') as file_handler:
        file_handler.write(data)

    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(result_file)

    if mime_type == 'image/jpeg':
        os.rename(result_file, result_file + '.jpg')
        result_file = result_file + '.jpg'
    elif mime_type == 'image/png':
        os.rename(result_file, result_file + '.png')
        result_file = result_file + '.png'
    elif mime_type == 'image/gif':
        os.rename(result_file, result_file + '.gif')
        result_file = result_file + '.gif'
    elif mime_type == 'image/bmp':
        os.rename(result_file, result_file + '.bmp')
        result_file = result_file + '.bmp'
    elif mime_type == 'image/tiff':
        os.rename(result_file, result_file + '.tiff')
        result_file = result_file + '.tiff'
    else:
        print('Not an image? %s' % mime_type)

    im=cv2.imread(result_file)
    #im=cv2.flip(im,1,1) #Flip to act as a mirror
    
    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini,1.05,5)# you may change params or 
    #remove them if faces not detected decrease them if u get falses then increase them
 
    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]

        # Close all started windows
        #cv2.destroyAllWindows()

        return labels_dict[label]
        #cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        #cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        #cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    