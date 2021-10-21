from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/mask', methods=['POST']) # , 'GET'
def detect_mask(img):
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(port=8888)