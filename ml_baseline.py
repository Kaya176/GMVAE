from __future__ import absolute_import, print_function
import torch

import argparse
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from utils.data_utils import *
from utils.perf_utils import *
from utils.reduc_utils import *
from utils.plot_utils import *

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.spatial import distance

import seaborn as sns
color = sns.color_palette()
import copy
import numpy as np

# import wandb
# wandb.init(project='21_kcc', entity='id4thomas')

ATK=1
SAFE=0

import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

label_vals={
    0:['BENIGN'],
    #Brute Force
    1:['SSH-Patator','FTP-Patator'],
    #DoS
    2:['DoS slowloris','DoS Slowhttptest','DoS Hulk','DoS GoldenEye'],
    #
    # 3:['Web Attack   Brute Force',]
    #Infiltration
    4:["Bot"],
    5:["DDoS"],
    6:["PortScan"],
    #ETC
    7:["Heartbleed","Infiltration"]

}

def make_cat_label(y,y_type,cat_dict):
    y[y_type!="BENIGN"]=3
    for y_cat in cat_dict.keys():
        for cat in cat_dict[y_cat]:
            y[y_type==cat]=y_cat
    return y

##### Load Data #####
data_path='data/cicids17/split/'
train_name='train'
x_train,y_train=get_hdf5_data(data_path+'{}.hdf5'.format(train_name),labeled=False)
y_train_type = np.load(data_path+'{}_label.npy'.format(train_name), allow_pickle=True)

y_train = np.zeros(x_train.shape[0])
# y_train[y_train_type!='BENIGN']=1
y_train=make_cat_label(y_train,y_train_type,label_vals)

# y_train = np.expand_dims(y_train.transpose(), axis=1)
# x_train = np.concatenate([x_train, y_train], axis=1)
print("Train: {}".format(x_train.shape))
print(y_train.shape)
#print("Train: Normal:{}, Atk:{}".format(x_train[y_train_type=='BENIGN'].shape[0],x_train[y_train_type!='BENIGN'].shape[0]))

#GET VALIDATION DATA!
data_path='data/cicids17/split/'
x_val,_=get_hdf5_data(data_path+'val.hdf5',labeled=False)
y_val_type=np.load(data_path+'val_label.npy',allow_pickle=True)

y_val=np.zeros(x_val.shape[0])
# y_val[y_val_type!='BENIGN']=1
y_val=make_cat_label(y_val,y_val_type,label_vals)

# y_val = np.expand_dims(y_val.transpose(), axis=1)
# x_val = np.concatenate([x_val, y_val], axis=1)
print("Val: Normal:{}, Atk:{}".format(x_val[y_val_type=='BENIGN'].shape[0],x_val[y_val_type!='BENIGN'].shape[0]))

x_test,_=get_hdf5_data(data_path+'test.hdf5',labeled=False)
y_test_type=np.load(data_path+'test_label.npy',allow_pickle=True)

y_test=np.zeros(x_test.shape[0])
# y_test[y_test_type!='BENIGN']=1
y_test=make_cat_label(y_test,y_test_type,label_vals)

print(y_test)
model=xgb.XGBClassifier()
# model=GaussianNB()
# model=RandomForestClassifier()
model.fit(x_train,y_train)

y_test_pred=model.predict(x_test)

prf(y_test,y_test_pred,avg_type='micro')

import pickle
# pickle.dump(model, open("cic_xgboost_imbalance.model", 'wb'))
pickle.dump(model, open("cic_xgboost.model", 'wb'))