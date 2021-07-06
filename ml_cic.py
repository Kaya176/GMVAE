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

from sklearn.metrics import fbeta_score, recall_score, matthews_corrcoef
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

# def make_cat_label(y,y_type,cat_dict):
#     y[y_type!="BENIGN"]=3
#     for y_cat in cat_dict.keys():
#         for cat in cat_dict[y_cat]:
#             y[y_type==cat]=int(y_cat)
#     return y

def make_cat_label(y,y_type,cat_dict):
    y[y_type!="BENIGN"]=1
    # for y_cat in cat_dict.keys():
    #     for cat in cat_dict[y_cat]:
    #         y[y_type==cat]=int(y_cat)
    return y

##### Load Data #####
data_path='data/cicids17/split/'
train_name='train'
x_train,y_train=get_hdf5_data(data_path+'{}.hdf5'.format(train_name),labeled=False)
y_train_type = np.load(data_path+'{}_label.npy'.format(train_name), allow_pickle=True)


y_train = np.zeros(x_train.shape[0])
# y_train[y_train_type!='BENIGN']=1
y_train=make_cat_label(y_train,y_train_type,label_vals)

sample_atk=False
if sample_atk:
    #Sample
    x_safe=x_train[y_train==0]
    x_atk=x_train[y_train==1]
    y_safe=np.zeros(x_safe.shape[0])
    y_atk=np.ones(x_atk.shape[0])

    #1 %
    sample_idx=np.random.choice(x_atk.shape[0], int(x_safe.shape[0]*0.01))
    x_atk_sampled=x_atk[sample_idx]
    y_atk_sampled=y_atk[sample_idx]

    x_train=np.concatenate((x_atk_sampled,x_safe),axis=0)
    y_train=np.concatenate((y_atk_sampled,y_safe),axis=0)

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
# model=xgb.XGBClassifier()
# # model=GaussianNB()
# # model=RandomForestClassifier()
# model.fit(x_train,y_train)

# y_test_pred=model.predict(x_test)
# print("XGBoost")
# print("Micro")
# prf(y_test,y_test_pred,avg_type='micro')

# print("Macro")
# prf(y_test,y_test_pred,avg_type='macro')

# import pickle
# # pickle.dump(model, open("cic_xgboost_imbalance.model", 'wb'))
# pickle.dump(model, open("cic_xgboost.model", 'wb'))

# # model=xgb.XGBClassifier()
# model=GaussianNB()
# # model=RandomForestClassifier()
# model.fit(x_train,y_train)

# y_test_pred=model.predict(x_test)
# print("Naive Bayes")
# print("Micro")
# prf(y_test,y_test_pred,avg_type='micro')

# print("Macro")
# prf(y_test,y_test_pred,avg_type='macro')

# import pickle
# # pickle.dump(model, open("cic_xgboost_imbalance.model", 'wb'))
# pickle.dump(model, open("cic_nb.model", 'wb'))

# # model=xgb.XGBClassifier()
# # model=GaussianNB()
# model=RandomForestClassifier()
# model.fit(x_train,y_train)

# y_test_pred=model.predict(x_test)
# print("Random Forest   ")
# print("Micro")
# prf(y_test,y_test_pred,avg_type='micro')

# print("Macro")
# prf(y_test,y_test_pred,avg_type='macro')

# import pickle
# # pickle.dump(model, open("cic_xgboost_imbalance.model", 'wb'))
# pickle.dump(model, open("cic_rf.model", 'wb'))


# Unsupervised
#OCSVM
from sklearn.svm import OneClassSVM
x_train_safe=x_train[y_train==0]

print("OCSVM RBF")
model=OneClassSVM(kernel='rbf')
model.fit(x_train_safe)
y_test_pred=model.predict(x_test)
y_test_pred[y_test_pred==1]=0
y_test_pred[y_test_pred<0]=1

print("Binary")
prf(y_test,y_test_pred,avg_type='binary')

print("Micro")
prf(y_test,y_test_pred,avg_type='micro')

print("Macro")
prf(y_test,y_test_pred,avg_type='macro')

#FPR
print("Normal")
cat_label=y_test[y_test==0]
cat_pred=y_test_pred[y_test==0]
print(accuracy_score(cat_label,cat_pred))
print("FPR",1-recall_score(cat_label,cat_pred,pos_label=0))

# print("OCSVM Linear")
# model=OneClassSVM(kernel='linear')
# model.fit(x_train_safe)
# y_test_pred=model.predict(x_test)
# y_test_pred[y_test_pred==1]=0
# y_test_pred[y_test_pred<0]=1

# print("Micro")
# prf(y_test,y_test_pred,avg_type='micro')

# print("Macro")
# prf(y_test,y_test_pred,avg_type='macro')

# #FPR
# print("Normal")
# cat_label=y_test[y_test==0]
# cat_pred=y_test_pred[y_test==0]
# print(accuracy_score(cat_label,cat_pred))
# print("FPR",1-recall_score(cat_label,cat_pred,pos_label=0))

# Isolation Forest
from sklearn.ensemble import IsolationForest
print("Isolation Forest")
model=IsolationForest(random_state=0).fit(x_train_safe)
y_test_pred=model.predict(x_test)
y_test_pred[y_test_pred==1]=0
y_test_pred[y_test_pred<0]=1

print("Binary")
prf(y_test,y_test_pred,avg_type='binary')

print("Micro")
prf(y_test,y_test_pred,avg_type='micro')

print("Macro")
prf(y_test,y_test_pred,avg_type='macro')


print("Normal")
cat_label=y_test[y_test==0]
cat_pred=y_test_pred[y_test==0]
print(accuracy_score(cat_label,cat_pred))
print("FPR",1-recall_score(cat_label,cat_pred,pos_label=0))
