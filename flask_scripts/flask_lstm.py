
import pandas as pd
from flask import Flask, jsonify, request
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import argparse
from lstm_price import create_dataset


app = Flask(__name__)
#model = load_model('/home/aarti/Downloads/all/lstmodel.h5')

'''with h5py.File('/home/aarti/Downloads/all/lstmodel.h5', 'r+') as f:
    if 'optimizer_weights' in f.keys():
        del f['optimizer_weights']
        f.close()
'''
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, 
	help="path of keras model to be loaded")

args = vars(ap.parse_args())
path = args["path"] 

model = load_model(path)

@app.route('/predict', methods=['POST'])
def apicall():
    
    try:
        test_json = request.get_json()
        #convert to pandas dataframe again
        data = pd.read_json(test_json, orient='records')
        
        
        trainy = np.reshape(data['SalePrice'].values, (len(data),1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_= scaler.fit_transform(trainy)
        look_back = 25
        trainX, trainY = create_dataset(data_, look_back=look_back)     # this function can be tweaked to get trainX only, remove trainY from return
        test = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    except Exception as e:
        raise e
    if test.empty:
        return(bad_request())
    else:
        
        prediction = model.predict(test)
        # inverse transform to get the correct form of prediction
        pred = scaler.inverse_transform(prediction)
        final_pred = {'Sale Price' :pred}
    
        responses = jsonify(final_pred)
        responses.status_code = 200
    
        return (responses)

@app.errorhandler(400)
def bad_request(error=None):
    message = {'status': 400,'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',}
    resp = jsonify(message)
    resp.status_code = 400
    return resp

if __name__=='__main__':
    app.run()
