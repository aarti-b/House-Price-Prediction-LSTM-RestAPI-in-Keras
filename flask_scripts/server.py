import json
import requests
import pandas as pd
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, 
	help="path of test csv file")

args = vars(ap.parse_args())
path = args["path"] 
 
#setting header to send and accept responses
def server_file():
    #Setting the headers to send and accept json responses
    header = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    #load csv data
    df  = pd.read_csv(path,encoding="utf-8-sig")
    #convert to JSON format
    data = df.to_json(orient='records')
    # post data to server
    resp = requests.post("http://127.0.0.1:5000/predict", data = json.dumps(data), headers = header)
    resp.status_code
    #get response as SalePrice
    resp.json()

if __name__=='__main__':
    server_file()
