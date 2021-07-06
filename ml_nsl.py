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

from data.preprocess_nsl import *
from sklearn.metrics import average_precision_score

cats=["DoS","U2R","R2L","Probe"]
sub_cats={
    'DoS':["neptune","smurf","pod","teardrop","land","back","apache2","udpstorm","processtable","mailbomb"],
    "U2R":["buffer_overflow","loadmodule","perl","rootkit","spy","xterm","ps","httptunnel","sqlattack","worm","snmpguess"],
    "R2L":["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","snmpgetattack","named","xlock","xsnoop","sendmail"],
    "Probe":["portsweep","ipsweep","nmap","satan","saint","mscan"]
}

# cat_tprs=[recall]
#     for cat in cats:
#         print(cat)
#         pred_cat=[]
#         y_cat=[]

#         for sub_cat in sub_cats[cat]:
#             # print(sub_cat)
#             pred_subcat=y_pred[y_test_types==sub_cat]
#             y_subcat=y_test[y_test_types==sub_cat]
            
#             pred_cat.append(pred_subcat)
#             y_cat.append(y_subcat)

#         pred_cat=np.concatenate(pred_cat,axis=0)
#         y_cat=np.concatenate(y_cat,axis=0)

#         print(pred_cat.shape)
#         print(accuracy_score(y_cat,pred_cat))
#         cat_tprs.append(accuracy_score(y_cat,pred_cat))

# def make_cat_label(y,y_type,cats,sub_cats):
#     for i in range(len(cats)):
#         cat=cats[i]
#         for sub_cat in sub_cats[cat]:
#             y[y_type==sub_cat]=i+1

#     return y

def make_cat_label(y,y_type,cats,sub_cats):
    return y

def filter_data(x,y,y_type,cats,sub_cats):
    filter_cat=['U2R','R2L']
    filter_cat=['U2R']
    for cat in filter_cat:
        for sub_cat in sub_cats[cat]:
            x=x[y_type!=sub_cat]
            y=y[y_type!=sub_cat]
            y_type=y_type[y_type!=sub_cat]
    return x,y,y_type

##### Load Data #####
def make_cat_label_type(y,y_type,cats,sub_cats):
    y_t=copy.deepcopy(y)
    cats=["DoS","Probe"]
    cats=["DoS","R2L","Probe"]
    for i in range(len(cats)):
        cat=cats[i]
        for sub_cat in sub_cats[cat]:
            y_t[y_type==sub_cat]=i+1

    return y_t

def load_data(cats,sub_cats):
    data_dir='data/nsl_kdd/split'
    train=pd.read_csv(data_dir+'/train.csv',header=None)
    val=pd.read_csv(data_dir+'/val.csv',header=None)
    test=pd.read_csv(data_dir+'/test.csv',header=None)

    service = open(data_dir+'/service.txt', 'r')
    serviceData = service.read().split('\n')
    service.close()

    flag = open(data_dir+'/flag.txt', 'r')
    flagData = flag.read().split('\n')
    flag.close()

    #Preprocess
    train_df,y_train,y_train_types,scaler,num_desc=preprocess(train,serviceData,flagData)  
    x_train=train_df.values
    # x_train,y_train=filter_label(x_train,y_train,select_label=SAFE)
    x_train,y_train,y_train_types=filter_data(x_train,y_train,y_train_types,cats,sub_cats)

    y_train=make_cat_label(y_train,y_train_types,cats,sub_cats)
    print("Train: Normal:{}, Atk:{}".format(x_train[y_train==0].shape[0],x_train[y_train!=0].shape[0]))

    val_df,y_val,y_val_types,_,_=preprocess(val,serviceData,flagData,is_train=False,scaler=scaler, num_desc=num_desc)
    x_val=val_df.values

    x_val,y_val,y_val_types=filter_data(x_val,y_val,y_val_types,cats,sub_cats)
    y_val=make_cat_label(y_val,y_val_types,cats,sub_cats)
    

    test_df,y_test,y_test_types,_,_=preprocess(test,serviceData,flagData,is_train=False,scaler=scaler, num_desc=num_desc)
    x_test=test_df.values

    x_test,y_test,y_test_types=filter_data(x_test,y_test,y_test_types,cats,sub_cats)
    y_test_t=make_cat_label_type(y_test,y_test_types,cats,sub_cats)
    y_test=make_cat_label(y_test,y_test_types,cats,sub_cats)

    return x_train,y_train,x_val,y_val,x_test,y_test,y_test_t

x_train,y_train,x_val,y_val,x_test,y_test,y_test_t=load_data(cats,sub_cats)
# sample_atk=True
# if sample_atk:
#     #Sample
#     x_safe=x_train[y_train==0]
#     x_atk=x_train[y_train==1]
#     y_safe=np.zeros(x_safe.shape[0])
#     y_atk=np.ones(x_atk.shape[0])

#     #1 %
#     sample_idx=np.random.choice(x_atk.shape[0], int(x_safe.shape[0]*0.01))
#     x_atk_sampled=x_atk[sample_idx]
#     y_atk_sampled=y_atk[sample_idx]

#     x_train=np.concatenate((x_atk_sampled,x_safe),axis=0)
#     y_train=np.concatenate((y_atk_sampled,y_safe),axis=0)



# print(y_test)
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


# cats=["normal","DoS","R2L","Probe"]
cats=["normal","DoS","R2L","Probe"]
# #FPR
# print("Normal")
# cat_label=y_test[y_test==0]
# cat_pred=y_test_pred[y_test==0]
# print(accuracy_score(cat_label,cat_pred))

# for i in range(1,len(cats)):
#     print(cats[i])
#     cat_label=y_test[y_test_t==i]
#     cat_pred=y_test_pred[y_test_t==i]

#     cat_label_one=np.ones_like(cat_label)
#     print(accuracy_score(cat_label_one,cat_pred))

# import pickle
# # pickle.dump(model, open("cic_xgboost_imbalance.model", 'wb'))
# pickle.dump(model, open("nsl_xgboost.model", 'wb'))

from sklearn.svm import SVC

model=SVC()
# model=GaussianNB()
# # model=RandomForestClassifier()
model.fit(x_train,y_train)

y_test_pred=model.predict(x_test)
print("XG")
print("Binary")
prf(y_test,y_test_pred,avg_type='binary')

print("Normal")
cat_label=y_test[y_test==0]
cat_pred=y_test_pred[y_test==0]
print(accuracy_score(cat_label,cat_pred))

for i in range(1,len(cats)):
    print(cats[i])
    cat_label=y_test[y_test_t==i]
    cat_pred=y_test_pred[y_test_t==i]

    cat_label_one=np.ones_like(cat_label)
    print(accuracy_score(cat_label_one,cat_pred))
exit()

model=xgb.XGBClassifier()
# model=GaussianNB()
# # model=RandomForestClassifier()
model.fit(x_train,y_train)

y_test_pred=model.predict(x_test)
print("XG")
print("Binary")
prf(y_test,y_test_pred,avg_type='binary')

print("Normal")
cat_label=y_test[y_test==0]
cat_pred=y_test_pred[y_test==0]
print(accuracy_score(cat_label,cat_pred))

for i in range(1,len(cats)):
    print(cats[i])
    cat_label=y_test[y_test_t==i]
    cat_pred=y_test_pred[y_test_t==i]

    cat_label_one=np.ones_like(cat_label)
    print(accuracy_score(cat_label_one,cat_pred))


# # model=xgb.XGBClassifier()
model=GaussianNB()
# # model=RandomForestClassifier()
model.fit(x_train,y_train)

y_test_pred=model.predict(x_test)
print("Naive Bayes")
print("Binary")
prf(y_test,y_test_pred,avg_type='binary')

print("Normal")
cat_label=y_test[y_test==0]
cat_pred=y_test_pred[y_test==0]
print(accuracy_score(cat_label,cat_pred))

for i in range(1,len(cats)):
    print(cats[i])
    cat_label=y_test[y_test_t==i]
    cat_pred=y_test_pred[y_test_t==i]

    cat_label_one=np.ones_like(cat_label)
    print(accuracy_score(cat_label_one,cat_pred))


# print("Micro")
# prf(y_test,y_test_pred,avg_type='micro')

# print("Macro")
# prf(y_test,y_test_pred,avg_type='macro')

# import pickle
# # pickle.dump(model, open("cic_xgboost_imbalance.model", 'wb'))
# pickle.dump(model, open("nsl_nb.model", 'wb'))

# # model=xgb.XGBClassifier()
# # model=GaussianNB()
model=RandomForestClassifier()
model.fit(x_train,y_train)

y_test_pred=model.predict(x_test)
print("Random Forest")
print("Binary")
prf(y_test,y_test_pred,avg_type='binary')

print("Normal")
cat_label=y_test[y_test==0]
cat_pred=y_test_pred[y_test==0]
print(accuracy_score(cat_label,cat_pred))

for i in range(1,len(cats)):
    print(cats[i])
    cat_label=y_test[y_test_t==i]
    cat_pred=y_test_pred[y_test_t==i]

    cat_label_one=np.ones_like(cat_label)
    print(accuracy_score(cat_label_one,cat_pred))
# print("Micro")
# prf(y_test,y_test_pred,avg_type='micro')

# print("Macro")
# prf(y_test,y_test_pred,avg_type='macro')

# import pickle
# # pickle.dump(model, open("cic_xgboost_imbalance.model", 'wb'))
# pickle.dump(model, open("nsl_rf.model", 'wb'))

#OCSVM
exit()

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

for i in range(1,len(cats)):
    print(cats[i])
    cat_label=y_test[y_test_t==i]
    cat_pred=y_test_pred[y_test_t==i]

    cat_label_one=np.ones_like(cat_label)
    print(accuracy_score(cat_label_one,cat_pred))

# print("OCSVM Linear")
# model=OneClassSVM(kernel='linear')
# model.fit(x_train_safe)
# y_test_pred=model.predict(x_test)
# y_test_pred[y_test_pred==1]=0
# y_test_pred[y_test_pred<0]=1

# print("Binray")
# prf(y_test,y_test_pred,avg_type='binary')

# print("Micro")
# prf(y_test,y_test_pred,avg_type='micro')

# print("Macro")
# prf(y_test,y_test_pred,avg_type='macro')

# cats=["normal","DoS","R2L","Probe"]

# #FPR
# print("Normal")
# cat_label=y_test[y_test==0]
# cat_pred=y_test_pred[y_test==0]
# print(accuracy_score(cat_label,cat_pred))

# for i in range(1,len(cats)):
#     print(cats[i])
#     cat_label=y_test[y_test_t==i]
#     cat_pred=y_test_pred[y_test_t==i]

#     cat_label_one=np.ones_like(cat_label)
#     print(accuracy_score(cat_label_one,cat_pred))

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

cats=["normal","DoS","R2L","Probe"]
#FPR
print("Normal")
cat_label=y_test[y_test==0]
cat_pred=y_test_pred[y_test==0]
print(accuracy_score(cat_label,cat_pred))

for i in range(1,len(cats)):
    print(cats[i])
    cat_label=y_test[y_test_t==i]
    cat_pred=y_test_pred[y_test_t==i]

    cat_label_one=np.ones_like(cat_label)
    print(accuracy_score(cat_label_one,cat_pred))

