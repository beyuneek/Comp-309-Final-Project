# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 00:28:13 2023

@author: Marco
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
import os
# Your API definition
app = Flask(__name__)
@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            scaled_df = scaler.fit_transform(query)
            # return to data frame
            query = pd.DataFrame(scaled_df, columns=model_columns)
            print(query)
            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to titanic model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    
    current_directory = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    lr = joblib.load(os.path.join(current_directory, 'model_lr.pkl')) # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load(os.path.join(current_directory, 'model_columns.pkl')) # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)
