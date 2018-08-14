# House-Price-Prediction-LSTM-RestAPI-in-Keras
AIM: To predict house price.
```
Requirements: 
Anaconda 3-5.2.0
Linux-Ubuntu 16.04
Python-3.6.6
keras=2.1.6
Tensorflow=1.9.0
Tensorflowjs=0.5.6
h5py=2.8.0
flask=1.0.2
numpy=1.14.1
pandas=0.23.4
matplotlib=2.2.3
argparse=1.1
seaborn=0.9.0
warnings
sklearn==0.19.2
math
json=2.0.9
requests=2.19.1
```
Method to execute code :

(1) HS_analysis.py
(2) lstm_price.py
(3) flask_lstm.py
(4) server.py

It is always better to create virtual environment and then execute prototype or any repository so as not to conflict with your system’s package’s configurations etc.

Method of creating Virtual Environment using conda :
```
conda update conda
conda create -n yourenvname python=x.x 
```
Then install packages you need to run repository
To activate run : ```source activate yourenvname```
To deactivate run : ```source deactivate```


(A) ----->  To get Analysis of data execute script (1). Type following command on shell:

```python HS_price_prediction.py --path /home/aarti/Downloads/all/train.csv --analysis True```

path to csv file is required. If you want final data then replace ```--analysis with --final_data```.

(B)-------> To train LSTM model (2) execute following :

```python lstm_price.py --path /path/to/trainfile  --h5path /path/where/h5/istostore --tfjspath /path/where/tfjstostore```

(C) -------> Make REST-API (3), execute following:

```python flask_lstm.py --path /path/to h5model```

a server would open up like :
Serving Flask app "flask_lstm" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
  * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

(D)---------->  Now run (4) using :

```python server.py --path /path/to/test/csvfile```
Or
Use ```curl``` along with data in ```-F``` flag to get Json prediction.

Note: Issue that I faced while runnning flask app to open server:
```UnsupportedOperation: not writable```

To resolve it :
You need to edit the ```echo``` function definition at ```../site-packages/click/utils.py``` the default value for the ```file``` parameter must be ```sys.stdout``` instead of ```None```.
Do the same for the ```secho``` function definition at ```../site-packages/click/termui.py```.
[reference](https://github.com/plotly/dash/issues/257)
