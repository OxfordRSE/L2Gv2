from tqdm.notebook import tqdm
import scipy.sparse as ss
import scipy.sparse.linalg as sl
import community

import torch_geometric as tg
import torch_scatter as ts
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import LargestConnectedComponents

import local2global as l2g
import local2global_embedding
from local2global_embedding import patches, clustering
from local2global_embedding.network import graph
from local2global_embedding.network import TGraph
from local2global import Patch
import local2global_embedding.embedding.svd as svd
import local2global_embedding.embedding.gae as gae
import local2global_embedding.patches as patches

import numpy as np 
import pandas as pd 
import torch 
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
#from google.colab import drive, files
import networkx as nx
import matplotlib.pyplot as plt


def raw_anomaly_score_node_patch(aligned_patch_emb, emb, node):
    return np.linalg.norm(aligned_patch_emb.get_coordinate(node)-emb[node])
    

def nodes_in_patches(patch_data):
    return [set(p.nodes.numpy()) for p in patch_data]

#is_in_patch=np.zeros((numb_nodes, numb_patches))
#for i in range(np.shape(emb)[0]):
    #for j in range(len(patch_data)):
        #is_in_patch[i,j]=i in nodes_in_patches[j]
        
def normalized_anomaly(patch_emb, patch_data, emb):
    nodes=nodes_in_patches(patch_data)
    numb_nodes=np.shape(emb)[0]
    numb_patches=len(patch_emb)
    st=np.zeros((numb_nodes, numb_patches))
    mu=np.zeros((numb_nodes, numb_patches))
    raw_anomaly=np.zeros((numb_nodes, numb_patches))


    for n in range(numb_nodes):
        for j in range(numb_patches):
            st[n, j]=np.std([raw_anomaly_score_node_patch(patch_emb[i], emb, n) for i in range(numb_patches) if (n in nodes[i]) & (i!=j)])
            mu[n, j]=np.mean([raw_anomaly_score_node_patch(patch_emb[i], emb, n) for i in range(numb_patches) if (n in nodes[i]) & (i!=j)])
            if n in nodes[j]:
                raw_anomaly[n,j]=raw_anomaly_score_node_patch(patch_emb[j], emb, n)
    final_score=np.zeros((numb_nodes, numb_patches))
    for n in range(numb_nodes):
        for j in range(numb_patches):
            if n in nodes[j]:
                if (st[n,j]!=0) & (str(mu[n,j])!= 'nan') & (str(st[n,j])!='nan'):
                    final_score[n,j]=(raw_anomaly[n,j]-mu[n,j])/st[n,j]

    return final_score

def get_outliers(patch_emb, patch_data, emb, k):
    out=[]
    numb_nodes=np.shape(emb)[0]
    numb_patches=len(patch_emb)
    final_score=normalized_anomaly(patch_emb, patch_data, emb)
    M=np.mean(final_score)
    S=np.std(final_score)
    for n in range(numb_nodes):
        for j in range(numb_patches):
            if np.abs(final_score[n,j]-M)>=k*S:
                out.append(n)
    return out

