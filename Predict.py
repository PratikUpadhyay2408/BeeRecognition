import os
import tensorflow as tf
import numpy as np

from skimage import io, transform
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# define a Flask app
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.isdir('uploads'):
    os.mkdir('uploads')
# define prediction parameters

IMAGE_SIZE = (100, 100)
list_lbl = ['Italian honey bee', '1 Mixed local stock 2', 'VSH (varroa sensitive hygiene) Italian honey bee',
            'Russian honey bee', 'Carniolan honey bee', 'Western honey bee']

# Upload the saved model
cnn_model = tf.keras.models.load_model(os.path.join('results', 'FinalModel.h5'))


def load_img(file, target_size):
    img = io.imread(file)
    img = transform.resize(img, target_size, mode='reflect')
    return img[:, :, :3]


def predict(file):
    img = load_img(file, target_size=IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    prob_array = cnn_model.predict(img)
    result = np.where(prob_array[0] == np.amax(prob_array[0]))[0][0]
    return list_lbl[result]



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']

        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        # make prediction about this image's class
        preds = predict(file_path)

        print('[PREDICTED CLASSES]: {}'.format(preds))

        return preds

    return None


if __name__ == '__main__':
    app.run(port=5000)
