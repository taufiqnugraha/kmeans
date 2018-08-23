
from flask import Flask, jsonify
import pymysql
from flask_cors import CORS

from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import requests
import json
from pylab import rcParams
from pandas import Series, DataFrame
from scipy.spatial.distance import cdist
from train_kmeans import kmeans

db = pymysql.connect('localhost','root', '', 'skripsi_pendidikan')
cursor = db.cursor()

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({"status": "success"})

@app.route('/api/getangkaharapan/', methods=['GET'])
def getipm():
    try:
        cursor.execute("SELECT kabupaten, MYS, IPM, tahun FROM pembangunan where tahun=2017")
        results = cursor.fetchall()
        data = []
        for item in results:
            data.append({'kabupaten': item[0], 'MYS': item[1], 'IPM': item[2],'tahun': item[3]})
        return jsonify(data)
    except Exception as e:
        return jsonify({'response': 404, 'error': e})
    
	
@app.route('/api/klustersummary/', methods=['GET'])
def getKluster():
    try:
        results = kmeans()
        return jsonify({'status_code': 200, 'result':json.loads(results.to_json(orient='records'))})
        # return jsonify({'kabupaten':result['kabupaten'].values,'cluster':result['cluster'].values})
    except Exception as e:
        return jsonify({'response': 404, 'error': e})

if __name__ == '__main__' : app.run(host="localhost", port=5000, debug=True)
