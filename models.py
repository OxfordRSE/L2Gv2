# %%
from tqdm import tqdm
import scipy.sparse as ss
import scipy.sparse.linalg as sl

import raphtory as rp
from raphtory import Graph as rgraph
import anomaly_detection
import os
import sqlite3
import nfts.dataset
import community

import torch 
from torch_geometric.utils.convert import from_networkx
import torch_geometric as tg
import torch_scatter as ts
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import to_networkx

from local2global_embedding import patches, clustering
from local2global_embedding.network import graph
from local2global_embedding.network import TGraph
import local2global as l2g
import local2global_embedding
from local2global import Patch
import local2global_embedding.embedding.svd as svd
import local2global_embedding.embedding.gae as gae
import local2global_embedding.patches as patches

import numpy as np 
import pandas as pd 
import optuna
from optuna.trial import TrialState
import polars as pl

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
#from google.colab import drive, files
import networkx as nx
import matplotlib.pyplot as plt

import glob
import os

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def speye(n, dtype=torch.float):
    """identity matrix of dimension n as sparse_coo_tensor."""
    return torch.sparse_coo_tensor(torch.tile(torch.arange(n, dtype=torch.long), (2, 1)),
                                   torch.ones(n, dtype=dtype),
                                   (n, n))


class VGAEconv(torch.nn.Module):
    def __init__(self, dim, num_node_features, hidden_dim=32, cached=True, bias=True, add_self_loops=True, normalize=True):
        super().__init__()
        self.conv1 = tg.nn.GCNConv(num_node_features, hidden_dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                   normalize=normalize)
        self.mean_conv2 = tg.nn.GCNConv(hidden_dim, dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                        normalize=normalize)
        self.var_conv2 = tg.nn.GCNConv(hidden_dim, dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                       normalize=normalize)

    def forward(self, data: tg.data.Data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        mu = self.mean_conv2(x, edge_index)
        sigma = self.var_conv2(x, edge_index)
        return mu, sigma


def train(data, model, loss_fun, num_epochs=100, verbose=True, lr=0.01, logger=lambda loss: None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    for e in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fun(model, data)
        loss.backward()
        optimizer.step()
        logger(float(loss))
        if verbose:
            print(f'epoch {e}: loss={loss.item()}')
        # schedule.step()
    return model

def VGAE_patch_embeddings(patch_data, dim=100, hidden_dim=32, num_epochs=100, decoder=None, device='cpu', lr=0.01):
    patch_list = []
    models = []
    for i, patch in enumerate(patch_data):
        
        
        
        if patch.x is None:
            patch.x = speye(patch.num_nodes)
        print(f"training patch {i} with {patch.edge_index.shape[1]} edges")   #added [i] to every patch
        model = tg.nn.VGAE(encoder=VGAEconv(dim, patch.x.shape[1], hidden_dim=hidden_dim), decoder=decoder).to(device)
        patch.to(device)

        def loss_fun(model, data):
            return model.recon_loss(model.encode(data), data.edge_index) + model.kl_loss() / data.num_nodes

        model = train(patch, model, loss_fun, num_epochs=num_epochs, lr=lr)
        with torch.no_grad():
            model.eval()
            coordinates = model.encode(patch).to('cpu').numpy()
            models.append(model)
            patch_list.append(l2g.Patch(patch.nodes.to('cpu').numpy(), coordinates))
    return patch_list, models

# %%


# %%
def Node2Vec_(data, emb_dim, w_length=20, c_size=10,
                              w_per_node=10, n_negative_samples=1, p=1, q=1, num_epochs=100):
    
    
    node2vec_model = Node2Vec(edge_index=data.edge_index, embedding_dim=emb_dim, walk_length=w_length, 
                                   context_size=c_size, walks_per_node=w_per_node,
                                   num_negative_samples=n_negative_samples,p=p, q=q).to(device)

# Optimizer
    optimizer = torch.optim.Adam(node2vec_model.parameters(), lr=0.01)

# Data loader for random walks
    loader = node2vec_model.loader(batch_size=128, shuffle=True, num_workers=0)

# Training loop
    

    node2vec_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec_model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader)}')

# Get embeddings after training
    node_embeddings = node2vec_model.embedding.weight.data.cpu()
    #models.append(model)
    
    return l2g.Patch(data.nodes.to('cpu').numpy(), node_embeddings)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def Node2Vec_patch_embeddings(patch_data, emb_dim , w_length=20, c_size=10,w_per_node=10, n_negative_samples=1, p=1, q=1, num_epochs=100):
    
    patch_list = []
   
    for i, data in enumerate(patch_data):
        print(f"training patch {i+1} with {data.edge_index.shape[1]} edges")  
        
        patch_list.append(Node2Vec_(data, emb_dim, w_length=w_length, c_size=c_size,
                              w_per_node=w_per_node, n_negative_samples=n_negative_samples, p=p, q=q))
    
    
    return patch_list


def chunk_embedding(chunk_size, patches, dim=2):  #this only work for Autonomous System dataset. 

    tAll=[from_networkx(g) for g in patches]
    list_set_nodes=[set(patches[i].nodes()) for i in range(len(patches))]

    nodes_in_intersection=set.intersection(*[set(gr) for gr in list_set_nodes])
    
    nodes_in_union=set()
    for s in list_set_nodes:
        nodes_in_union = nodes_in_union.union(s)
    
    missing_nodes=set(i for i in range(max(nodes_in_union)))
    for i in range(max(nodes_in_union)):
        if i in nodes_in_union:
            missing_nodes.remove(i)
    
    for i, p in tqdm(enumerate(tAll)):
        p.nodes=torch.Tensor(list(patches[i].nodes)).int()
        p.num_nodes=p.nodes.size(0)
    sub_patches = [tAll[i:i + chunk_size] for i in range(0, len(tAll), chunk_size)]

    nodes_in_intersection=[]
    list_nodes_tot=[]
    for g in tqdm(sub_patches):
    
        nodes_in_each_p=[set(p.nodes.tolist()) for p in g]
        nodes_in_intersection.append(set.intersection(*nodes_in_each_p))
        nodes_tot=[]
        for p in nodes_in_each_p:
            nodes_tot+=p
        list_nodes_tot.append(set(nodes_tot))  

    emb=[]
    Ppatch=[]
    for i, As in tqdm(enumerate(sub_patches)):
        p_emb=VGAE_patch_embeddings(As, dim=dim, hidden_dim=32, num_epochs=100, decoder=None, device='cpu', lr=0.01)
    
        prob = l2g.utils.WeightedAlignmentProblem(p_emb[0])  #embedding of the full graph using embeddings of each patch
    
        emb.append(prob.get_aligned_embedding())
    Ppatch2=[Patch( list(list_nodes_tot[i]), emb[i][list(list_nodes_tot[i])]) for i in range(len(emb))]

    Prob=l2g.utils.WeightedAlignmentProblem(Ppatch2)

    Emb=Prob.get_aligned_embedding()

    Nodes_emb=Emb[list(nodes_in_union)]

    return Nodes_emb, Prob
# %%
