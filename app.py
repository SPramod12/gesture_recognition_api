from flask import Flask, request
import cv2
from keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
import os, psutil
import gc

app = Flask(__name__)

def preprocess(video, start=6, end=24):
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    img_start=0
    flag=False
    x = np.zeros((18, 112, 112, 3))
    while success:
        if count==start:
            flag=True
        if count==end:
            flag=False
        if flag:
            image = cv2.rotate(image, cv2.ROTATE_180)
            # image = image[360:-360,20:-20,:]
            image = cv2.resize(image, (112, 112))
            x[img_start,:,:,:] = image
            img_start = img_start+1
        success,image = vidcap.read()
        count += 1
    vidcap.release()
    if os.path.exists(video):
        os.remove(video)
    else:
        print("The file does not exist")
    gc.collect()
    return x

def prediction(file_path):
    model = load_model('models/model-00005-0.38287-0.87783-0.62432-0.85000.h5')
    x = preprocess(file_path).reshape(1, 18, 112, 112,3)
    pred = model.predict(x)[0].argmax()
    gc.collect()
    del model
    del x
    return pred

@app.route('/', methods=['GET'])
def index():
    return "working"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        gestures = ['Left Swipe', 'Right Swipe', 'Stop', 'Tumbs Down', 'Tumbs Up']
        f = request.files['video']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)
        gc.collect()
        final = gestures[prediction(file_path)]
        del gestures
        del f
        return final
    else:
        return "None"

if __name__ == '__main__':
    app.run(debug=True)
