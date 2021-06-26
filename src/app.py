import pandas as pd
import numpy as np
import hydra
import os
import json
import yaml
import pickle
import time

from flask import Flask, request, jsonify
import mysql.connector
import prediction

#26.6.2021
#1. Change the db into production db
#2. Deploy into AWS EC2
#3. Deploy the model into AWS EKS

app = Flask(__name__)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="asdf1234",
    database="water_quality"
)

@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        if request.json:
            mycursor = db.cursor()

            pH  = request.json["ph"]
            Hardness = request.json["Hardness"]
            Solids =    request.json["Solids"]
            Chloramines =     request.json["Chloramines"]
            Sulfate =     request.json["Sulfate"]
            Conductivity =    request.json["Conductivity"]
            Organic_carbon =     request.json["Organic_carbon"]
            Trihalomethanes =    request.json["Trihalomethanes"]
            Turbidity =    request.json["Turbidity"]
            predict_result = prediction.api_response(request.json)

            mycursor.execute("""INSERT INTO water_q(
                            pH, Hardness, Solids, Chloramines, Sulfate, Conductivity,
                            Organic_carbon, Trihalomethanes, Turbidity, Potability) 
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity,
                            Organic_carbon, Trihalomethanes, Turbidity, predict_result))

            db.commit()
            response = {"Prediction": int(predict_result)}
            return jsonify(response)
    else:
        return None
        
if __name__ == "__main__":
    app.run(port = 5001, debug = True)