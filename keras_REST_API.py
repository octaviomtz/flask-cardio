from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
import tensorflow as tf
import base64
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
import io
import json

K.clear_session()

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None

def load_model():
    # model trained using: "polarmaps keras model16B- pretrained 1 inputs - ...
    # intermediate - resample7 - testset-n_test-Delete
    global model
    # get model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # get weights
    model.load_weights('best_model16.hdf5')
    # According to https://github.com/keras-team/keras/issues/2397
    global graph
    graph = tf.get_default_graph()

def prepare_image(image):
    # resize the input image and preprocess it
    # jpgfile = load_img(image)
    # image_array=img_to_array(jpgfile)
    image=np.pad(image,[(0, 1), (0, 1),(0, 0)],mode='constant')
    image=np.expand_dims(image,0)
    image=image/255
    # return the processed image
    return image

@app.route('/')
def home():
	return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    response = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        message = request.get_json(force=True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        a=io.BytesIO(decoded)
        image = Image.open(a)
        image = prepare_image(image)
        with graph.as_default():
                preds = model.predict(image)
        #print(preds)
        
        #response = {"predictions":str(preds[0][0])}
        response["predictions"] = str(preds[0][0])

        # indicate that the request was a success
        response["success"] = True
        print(response)
        print(jsonify(response))

    # return the data dictionary as a JSON response
    return jsonify(response)
    #return json.dumps({'status':'OK','user':str(preds)})
    

load_model()
app.run()