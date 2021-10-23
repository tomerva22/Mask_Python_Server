from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/mask', methods=['POST'])
def detect_mask():
    data = request.data
    print(data)
    return jsonify({'Hello': 'world'})
