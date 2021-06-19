import json
import yaml
import pickle
import numpy as np

'''1. Exception Handling'''

class NotInCols(Exception):
    def __init__(self, message="Incorrect Columns!"):
        self.message = message
        super().__init__(self.message)

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)


#2. Read the Json File
def read_schema(schema_path = 'schema.json'):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

#3. Read the model from config.yaml file

def read_params(config_path = '../config.yaml'):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

#4. Validate the columns name & value in the range

def validation_input(dict_request):
    def validation_cols(col):
        schema = read_schema()
        schema_keys = schema.keys()
        if col not in schema_keys:
                raise NotInCols

    def validation_value(col, value):
        schema = read_schema()
        if not(schema[col]["min"] < float(dict_request[col]) < schema[col]["max"]):
            raise NotInRange

    for col, value in dict_request.items():
        validation_cols(col)
        validation_value(col, value)

    return True

#5. Load the config.yaml & selected model to predict the data:

def api_response(dict_request):
    config = read_params('config.yaml')
    model_dir_path = config["selected_model"]
    with open(model_dir_path, 'rb') as file:
        model = pickle.load(file)
    try:
        if validation_input(dict_request):
            data = np.array([list(dict_request.values())])
            prediction = model.predict(data).tolist()[0]
            #response = {'Prediction Result': int(prediction)}
            return prediction
    except NotInCols as e:
        response = {"the_expected_colums_name": read_schema(), "response": str(e) }
        return response   

    except NotInRange as e:
        response = {"the_expected_value_range": read_schema(), "response": str(e) }
        return response

    except Exception as e:
        response = {"The expected_range": read_schema(),"response": str(e) }
        return response
if __name__ == "__main__":
    api_response({
    "ph": 7,
    "Hardness": 180,
    "Solids": 20000,
    "Chloramines": 7,
    "Sulfate": 200,
    "Conductivity": 500,
    "Organic_carbon": 13,
    "Trihalomethanes": 66,
    "Turbidity": 4
})