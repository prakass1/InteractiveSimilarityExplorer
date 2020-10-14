import pickle
import properties

'''
Save the data objects in model location, save data in data location, and load as well
'''

def save_model(model_name, obj):
    print("Saving model to the location ", properties.model_location)
    model_file = open("".join(properties.model_location + model_name), "wb")
    # Save the model
    pickle.dump(obj, model_file)

def save_ts_model(model_name, obj):
    print("Saving model to the location ", "ts_models/")
    model_file = open("".join("ts_models/" + model_name), "wb")
    # Save the model
    pickle.dump(obj, model_file)

def load_ts_model(model_name):
    model = pickle.load(open("ts_models/" + model_name, 'rb'))
    return model


def load_model(model_name):
    model = pickle.load(open(properties.model_location + model_name, 'rb'))
    return model


import json
def save_data(model_name, obj):
    print("Saving data to the location ", properties.data_location)
    with open("".join(properties.data_location + model_name + ".json"), 'w') as f:
        json.dump(obj, f)

def load_data(model_name):
    f = open("".join(properties.data_location + model_name + ".json"))
    data_question = json.load(f)
    return data_question
