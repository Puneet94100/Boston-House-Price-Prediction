import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


# defining the flask app
app=Flask(__name__)

#load the model and standard scaler
regmodel=pickle.load(open('regression_model.pkl','rb'))
scaler=pickle.load(open('standard_scaler.pkl','rb'))

# creaqte the route
@app.route('/', methods=['GET','POST'])

# defining a function that render the template as we hit on 
# flask app it will redirect to home.html page
def home():
    return render_template('home.html')

# create api that gives the prediction
# we will give the input, it will capture the input
# and model will takes and model will gives prediction on this data
@app.route('/predict_api', methods=['POST'])
def predict_api():

# As we hit this api, the input we give in json formate and it will be captured 
# in 'data' key after that the information that is present in data key
# we will capture it using request.json and store it in data variable
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    # We have to standardization before sening the data to model
    # The data is in key and value format, we need only value, and store 
    # the values into list and after that reshape the input using array
    # because standard scaler needs 2d array

    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))

    # preiction using regmodel
    output=regmodel.predict(new_data)
    
    # The output we will find in array but we need only the value
    # So we are doing indexing
    print(output[0])
    return jsonify(output[0])


# creating a methods that takes values from form and we use these values
# to make the prediction by model

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in  request.form.values()]
    # standardize the data
    new_data=scaler.transform(np.array(data).reshape(1,-1))
    output=regmodel.predict(new_data)[0]

    # render to specific html
    # keeping a place holder that is holding the prediction value
    return render_template('home.html',prediction_text=f'The House price prediction is {output}')



if __name__=="__main__":
    app.run(debug=True)