from __future__ import absolute_import, print_function
import torch

import argparse
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

# from model.ae import AE,AE_split_train
#from model.classifier import CF

from utils.data_utils import *
from utils.perf_utils import *
from utils.reduc_utils import *
from utils.plot_utils import *

from data.preprocess_nsl import *

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.spatial import distance

import seaborn as sns
color = sns.color_palette()
import copy
import numpy as np

#GMVAE
import torch
# from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from model.GMVAE import *

import copy

from sklearn.metrics import fbeta_score, recall_score, matthews_corrcoef

ATK=1
SAFE=0

# Argument Setting
# parser = argparse.ArgumentParser()

# parser.add_argument("--seed", default=42, type=int,
#                     help="random seed for reproductability")

# #Model Config
# parser.add_argument("--l_dim", default=32, type=int,
#                     help="Latent Dim")
# parser.add_argument("--num_layers", default=2, type=int,
#                     help="number of layers")
# parser.add_argument("--size", default=64, type=int,
#                     help="Smallest Hid Size")
# #Regularization
# parser.add_argument("--do", default=0, type=float,
#                     help="dropout rate")
# parser.add_argument("--bn", default=0, type=int,
#                     help="batch norm: 1 to use")

# parser.add_argument("--epoch", default=10, type=int,
#                     help="training epochs")
# parser.add_argument("--batch_size", default=8192, type=int,
#                     help="batch size for train and test")
# parser.add_argument("--lr", default=1e-4, type=float,
#                     help="learning rate")

# parser.add_argument("--data", default="cic", type=str,
#                         help="Dataset")


#GMVAE Args
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                    help='Path for input file. First line should contain number of lines to search in')

## Dataset
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['mnist'],
                    default='mnist', help='dataset (default: mnist)')
parser.add_argument('--seed', type=int, default=10, help='random seed (default: 0)')

## GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

## Training
parser.add_argument('--epochs', type=int, default=20,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=8192, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=8192, type=int,
                    help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay_epoch', default=-1, type=int, 
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')

## Architecture
parser.add_argument('--num_classes', type=int, default=2,
                    help='number of classes (default: 10)')
parser.add_argument('--gaussian_size', default=32, type=int,
                    help='gaussian size (default: 64)')
parser.add_argument('--input_size', default=114, type=int,
                    help='input size (default: 784)')

## Partition parameters
parser.add_argument('--train_proportion', default=1.0, type=float,
                    help='proportion of examples to consider for training only (default: 1.0)')

## Gumbel parameters
parser.add_argument('--init_temp', default=1.0, type=float,
                    help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
parser.add_argument('--decay_temp', default=1, type=int, 
                    help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
parser.add_argument('--hard_gumbel', default=0, type=int, 
                    help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
parser.add_argument('--min_temp', default=0.5, type=float, 
                    help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)' )
parser.add_argument('--decay_temp_rate', default=0.013862944, type=float,
                    help='Temperature decay rate at every epoch (default: 0.013862944)')

## Loss function parameters
parser.add_argument('--w_gauss', default=1, type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_categ', default=1, type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='bce', help='desired reconstruction loss function (default: bce)')

## Others
parser.add_argument('--verbose', default=1, type=int,
                    help='print extra information at every epoch.(default: 0)')
parser.add_argument('--random_search_it', type=int, default=20,
                    help='iterations of random search (default: 20)')

args = parser.parse_args()



# Fix seed
set_seed(args.seed)
device = torch.device('cuda:0')

#Set Labels
cats=["DoS","U2R","R2L","Probe"]
sub_cats={
    'DoS':["neptune","smurf","pod","teardrop","land","back","apache2","udpstorm","processtable","mailbomb"],
    "U2R":["buffer_overflow","loadmodule","perl","rootkit","spy","xterm","ps","httptunnel","sqlattack","worm","snmpguess"],
    "R2L":["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","snmpgetattack","named","xlock","xsnoop","sendmail"],
    "Probe":["portsweep","ipsweep","nmap","satan","saint","mscan"]
}


# def make_cat_label(y,y_type,cats,sub_cats):
#     cats=["DoS","Probe"]
#     cats=["DoS","R2L","Probe"]
#     for i in range(len(cats)):
#         cat=cats[i]
#         for sub_cat in sub_cats[cat]:
#             y[y_type==sub_cat]=i+1

#     return y

def make_cat_label(y,y_type,cats,sub_cats):
    return y

def make_cat_label_type(y,y_type,cats,sub_cats):
    y_t=copy.deepcopy(y)
    cats=["DoS","Probe"]
    cats=["DoS","R2L","Probe"]
    for i in range(len(cats)):
        cat=cats[i]
        for sub_cat in sub_cats[cat]:
            y_t[y_type==sub_cat]=i+1

    return y_t

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
print(y_val[y_val>1])
#Dataset
class CICDataset(Dataset):
    def __init__(self,x,y,num_classes=2):
        #Numpy -> Torch
        self.x=torch.from_numpy(x).float()
        self.y=torch.from_numpy(y).long()

        #Y to Onehot
        self.y_oh=torch.nn.functional.one_hot(self.y,num_classes=num_classes).float()
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx],self.y_oh[idx]

#Load to Cuda
train_dataset=CICDataset(x_train,y_train,num_classes=args.num_classes)
val_dataset=CICDataset(x_val,y_val,num_classes=args.num_classes)
test_dataset=CICDataset(x_test,y_test,num_classes=args.num_classes)

train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False)
test_loader=DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)


#Get Model
# layers=[args.l_dim]
# for i in range(0,args.num_layers):
#     #Multiplying
#     # layers.append(args.l_dim*2**(i))
#     #Fixed
#     layers.append(args.size*2**(i))
# layers.reverse()
# layers=
# model_config={
#     'd_dim':80,
#     'layers':layers
# }
# model_desc='AE-{}_{}_{}'.format(args.size,args.l_dim,args.num_layers)
# model=AE_split_train(model_config).to(device)


model= GMVAE(args)

# wandb.watch(model)

history_loss = model.train(train_loader, val_loader,test_loader)

#Save Model
torch.save(model.network.state_dict(), f"./weights/nsl_{args.gaussian_size}_{args.epochs}.pt")

test_latent=model.latent_features(test_loader)
print(test_latent)
print(test_latent.shape)
accuracy, nmi = model.test(test_loader)
print("Testing phase...")
print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi))

#Load Model
# model= GMVAE(args)
# model.network.load_state_dict(torch.load(f"./weights/nsl_{args.gaussian_size}_{args.epochs}.pt"))
# accuracy, nmi = model.test(test_loader)
# print("Testing phase...")
# print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi))



test_acc,y_test_pred=model.pred_label(test_loader)

#FPR
print("Normal")
cat_label=y_test[y_test==0]
cat_pred=y_test_pred[y_test==0]

print("FPR",1-recall_score(cat_label,cat_pred,pos_label=0))

cats=["normal","DoS","R2L","Probe"]
for i in range(1,len(cats)):
    print(cats[i])
    cat_label=y_test[y_test==i]
    cat_pred=y_test_pred[y_test==i]

    cat_label_one=np.ones_like(cat_label)
    print(accuracy_score(cat_label_one,cat_pred))
