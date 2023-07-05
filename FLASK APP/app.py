from flask import Flask, render_template, url_for,request
import pickle as p
import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



modelfile = 'models/final_prediction.pickle'  
model = p.load(open(modelfile, 'rb'))
scaler= pickle.load(open('models/scaler.pickle','rb'))
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html') 

@app.route('/predict',methods =['GET','POST'])
def predict():
    Channel = float(request.form["Channel"])
    Region =float(request.form['Region'])
    Fresh = float(request.form['Fresh'])
    Milk=float(request.form['Milk'])
    Grocery = float(request.form['Grocery'])
    Frozen  = float(request.form['Frozen'])
    Detergents_Paper= float(request.form['Detergents_Paper'])
    Delicassen= float(request.form['Delicassen'])

    total = [[Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen]]
    prediction = model.predict(scaler.transform(total))
    prediction = int(prediction[0])

    if prediction==0:
        return render_template('index.html',predict="Customer Belongs to Cluster Label 0")
    
    if prediction==1:
        return render_template('index.html',predict="Customer Belongs to Cluster Label 1")
    if prediction==2:
        return render_template('index.html',predict="Customer Belongs to Cluster Label 2")
    
    if prediction==3:
        return render_template('index.html',predict="Customer Belongs to Cluster Label 3")
    else:
        return render_template('index.html',predict="Customer Belongs to Cluster Label 4")       




if __name__ == '__main__':
    app.run(debug=True)
