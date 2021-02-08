from flask import Flask, render_template, render_template_string, request
import os
import keras
import numpy as np
import seaborn as sns
from warnings import warn
import warnings
warnings.simplefilter("ignore")
from tensorflow.keras import backend
#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
#tensorflow.keras.__version__
#from tensorflow.keras.models import load_model
from keras.models import load_model
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Dense

class Object(object):
    pass

tfback._SYMBOLIC_SCOPE = Object()
tfback._SYMBOLIC_SCOPE.value = True

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).
    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
tfback._get_available_gpus()
#tf.config.experimental_list_devices()
tf.config.list_logical_devices()


global graph
graph = tf.compat.v1.get_default_graph()
app = Flask(__name__)

#model_path = os.path.join(os.getcwd(), ".", "model/model-v1.h5")
model_path = os.path.join(os.getcwd(), ".", "model/model.h5")
model = keras.models.load_model(model_path)
#model = load_model(model_path)
model.summary()

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = np.array(request.form['img'].split(","), np.float32).reshape(1, 28, 28, 1)
    img /= 255
    #print(img)
    result = model.predict(img)[0]
    return render_template_string(",".join(result.astype(str)))
    # return render_template_string("0.0985102057457,0.0936895906925,0.0623816810548,0.102438829839,0.108508199453,0.180629864335,0.0768260806799,0.0812399238348,0.0883773490787,0.107398249209")

if __name__ == '__main__':
    app.run(debug=True, threaded=False, use_reloader=True)