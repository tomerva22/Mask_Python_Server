from flask import Flask, jsonify, request
from flask.helpers import make_response
from flask.wrappers import Response
from flask_cors import CORS
import cv2
import os
import magic
import numpy as np
import tensorflow as tf
import random

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
file_num = 0

@app.route('/mask', methods=['POST'])
def detect_mask():
    data = request.data
    model = tf.keras.models.load_model("mobileNetV2FaceMask.model")
    labels_dict={0:'without_mask',1:'with_mask'}
    file_num = random.randint(0,10000)
    print(file_num)
    result_file = "newfile" + str(file_num)
    file_num+=1
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
    im=cv2.flip(im,1,1) #Flip to act as a mirror
    im =  im.reshape(-1, im.shape[0], im.shape[1], im.shape[2])
    normalized=im/255.0
    result=model.predict(normalized)
    label = np.argmax(result,axis=1)[0]
    print(labels_dict[label])
    response = make_response(labels_dict[label])
    os.remove(result_file)
    print(response)
    return response


if __name__ == '__main__':
    app.run(port=8888)