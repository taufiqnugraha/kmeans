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

def kmeas():
    data = requests.get('http://localhost:8000/api/tingkatbutuhguru').json() 
    sekolah = []
    siswa = []
    guru = []
    wilayah = []
    all_data = [] 
    cluster = data['results']
    for val in cluster:
        sekolah.append(int(val['sekolah']))
        siswa.append(int(val['siswa']))
        guru.append(int(val['guru']))
        wilayah.append(val['wilayah'])
        all_data.append({"sekolah":int(val['sekolah']),"siswa":int(val['siswa']),"guru":int(val['guru'])})
    df = pd.DataFrame(all_data)
    df_wilayah = pd.DataFrame(wilayah)
    df.describe()
    clustervar=df.copy()
    clustervar['guru']=preprocessing.scale(clustervar['guru'].astype('float64'))
    clustervar['siswa']=preprocessing.scale(clustervar['siswa'].astype('float64'))
    clustervar['sekolah']=preprocessing.scale(clustervar['sekolah'].astype('float64'))
    clus_train = clustervar
    clusters=range(1,11)
    meandist=[]
    for k in clusters:
        model=KMeans(n_clusters=k)
        model.fit(clus_train)
        clusassign=model.predict(clus_train)
        meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))/clus_train.shape[0])
    # plt.plot(clusters, meandist)
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Average distance')
    # plt.title('Selecting k with the Elbow Method')
    model3=KMeans(n_clusters=3)
    model3.fit(clus_train)
    clusassign=model3.predict(clus_train)
    plt.xlabel('Canonical variable 1')
    plt.ylabel('Canonical variable 2')
    plt.title('Scatterplot of Canonical Variables for 2 Clusters')
    clus_train.reset_index(level=0, inplace=True)
    cluslist=list(clus_train['index'])
    labels=list(model3.labels_)
    newlist=dict(zip(cluslist, labels))
    newclus= pd.DataFrame.from_dict(newlist, orient='index')
    newclus.columns = ['cluster']

    newclus.reset_index(level=0, inplace=True)
    merged_train=pd.merge(clus_train, newclus, on='index')
    merged_train.head(n=100)
    merged_train.cluster.value_counts()

    result1 = pd.concat([df,df_wilayah,newclus], axis=1)
    result = result1.rename(columns = {0 : 'kabupaten'})

    return result