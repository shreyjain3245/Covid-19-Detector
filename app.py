import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = load_model('model_CNN.h5')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index1.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        name=request.form["name"]
        img= request.files["image"]
        print(img)
        
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(basepath, secure_filename(img.filename))
        img.save(img_path)
    

        # Make prediction
        
        img = image.load_img(img_path, target_size=(224, 224))
        img=image.img_to_array(img)
        img=img/255
        img=np.expand_dims(img,axis=0)
        
        p=(model.predict(img)> 0.5).astype("int32") # Since sigmoid function
        print(p)
        try:    
            if p[0][0]==0:
              s=name+" is diagnosed with Covid-19."  
              return render_template('index1.html',prediction_text=s,image1=img_path);
            elif p[0][0]==1:
              s=name+" is Normal."
              return render_template('index1.html',prediction_text=s);

        except:
             return render_template('index1.html',prediction_text="Upload an X-Ray image")
    else:
        return render_template('index1.html')

if __name__=="__main__":
    app.run(debug=True)
