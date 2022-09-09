from flask import Flask, request, jsonify, send_file
import cv2
from keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
import os
import gc
from flask_cors import CORS
import glob
# import imageio
# from pathlib import Path

app = Flask(__name__)
CORS(app)

def preprocess(video, start=6, end=24):
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    img_start=0
    flag=False
    X = np.zeros((18, 112, 112, 3))
    img_list = []
    while success:
        if count==start:
            flag=True
        if count==end:
            flag=False
        if flag:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.rotate(image, cv2.ROTATE_180)
            image = image[160:,:,:]
            image = cv2.resize(image, (112, 112))
            img_list.append(image)
            image = (image-image.min())/(image.max()-image.min())
            X[img_start,:,:,:] = image
            img_start = img_start+1
        success,image = vidcap.read()
        count += 1
    vidcap.release()
    # p = ''.join(os.path.split(Path(video))[:-1])
    # p = os.path.join(p, 'video.gif')
    # imageio.mimsave(p, img_list, fps=10)
    # gif = send_file(p,as_attachment=True)
    if os.path.exists(video):
        os.remove(video)
    gc.collect()
    return X

def prediction(file_path):
    model = load_model('models/model-00005-0.38287-0.87783-0.62432-0.85000.h5')
    x = preprocess(file_path).reshape(1, 18, 112, 112,3)
    pred = model.predict(x)[0]
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
        del_dir = os.path.join(basepath, 'uploads')
        del_files = glob.glob(del_dir+"/*")
        if os.path.exists(del_dir):
            for df in del_files:
                os.remove(df)
        if not os.path.exists(del_dir):
            os.makedirs(del_dir)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        gc.collect()
        prob = prediction(file_path)
        final = gestures[prob.argmax()]
        prob = [str(i) for i in prob]
        prob_dict = dict(zip(gestures, prob))
        del gestures
        del f
        print(final)
        return jsonify({'prediction':final, 'probs':prob_dict})
    else:
        return "None"

if __name__ == '__main__':
    app.run(debug=True)
