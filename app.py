# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:34:20 2020

@author: Kaustav Chanda
"""


import tensorflow as tf
import tensorflow_hub as hub


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'text.h5'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        result = request.form['user_review']
        review = request.form['user_review']
        review = tf.convert_to_tensor([review])
        result = model.predict_classes(review)[0][0]
        if(result==0):
            result = 'This is a negative review'
            tone = 'danger'
        else:
            result = 'This is a positive review'
            tone = 'success'
    return render_template('index.html',review = request.form['user_review'],result=result,tone= tone)


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('index.html')

if __name__ == '__main__':
    model = tf.keras.models.load_model(MODEL_PATH,custom_objects={'KerasLayer':hub.KerasLayer})
    model.summary()
    model._make_predict_function()  
    print("Loaded model")
    app.run(debug=True)

