import pandas as pd
import numpy as np
import hydra
import os
import json
import yaml
import pickle
import time

from flask import Flask, request, jsonify
#from flask_mysqldb import MySQL
import prediction

#19.6.2021
#1. Write the json request & prediction into AWS RDS MySQL
#2. Deploy into AWS EC2

app = Flask(__name__)

#db = yaml.load('../db.yaml',  Loader = yaml.FullLoader)
#app.config["MYSQL_HOST"] = db["mysql_host"]
#app.config['MYSQL_USER'] = db['mysql_user']
#app.config['MYSQL_PASSWORD'] = db['mysql_password']
#app.config['MYSQL_DB'] = db['mysql_db']
#mysql = MySQL(app)

@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        if request.json:
            response = prediction.api_response(request.json)
            return jsonify(response)
    else:
        return None
        
if __name__ == "__main__":
    app.run(port = 5001, debug = True)