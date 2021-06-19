import pandas as pd
import numpy as np
import hydra
import os
import json
import yaml
import pickle
import time

from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
import prediction

#19.6.2021
#1. Write the json request & prediction into AWS RDS MySQL
#2. Deploy into AWS EC2

app = Flask(__name__)

def read_params(config_path = '../config.yaml'):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        if request.json:
            response = prediction.api_response(request.json)
            return jsonify(response)
            #ph = request.json["ph"]
            #return jsonify(ph)
    else:
        return None
        
if __name__ == "__main__":
    app.run(port = 5001, debug = True)