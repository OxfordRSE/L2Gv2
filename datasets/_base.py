import logging
from tqdm import tqdm
import os
from os import path
import numpy as np
import polars as pl
import torch
import networkx as nx
import raphtory as rp
from raphtory import Graph as rGraph
import sqlite3
import nfts.dataset
from torch_geometric.data import Data

PATH = path.join(path.dirname(__file__), 'data/')
#PATH = './data/'
DATASETS = {'AS': ['snap-as/as_edges.parquet'], 
            'elliptic': ['elliptic/elliptic_edges.parquet',
                         'elliptic/elliptic_nodes.parquet'], 
            'nfts': ['nfts/nfts_edges.parquet'],
            'nAS': ['nas/nas_edges.parquet',
                    'nas/nas_nodes.parquet']}
FORMATS = ['networkx', 'tgeometric', 'raphtory', 'polars', 'edge_list']
EDGE_COLUMNS = ['source', 'dest'] # required columns
NODE_COLUMNS = ['nodes'] # required column for nodes

class DataLoader:
    """Take a dataframe representing a (temporal) graph and provide 
    methods for loading the data in different formats. 
    """

    def __init__(self, 
                 source: str='AS', 
                 datasets: dict[str]=DATASETS,
                 path: str=PATH):
        assert source in datasets
        files = datasets[source]
        files = [path+f for f in files]
        self.__load_files(files)
        
    def __load_files(self, files):
        edgefile = None
        nodefile = None
        for f in files:
            if 'edges' in f:
                edgefile = f 
            if 'nodes' in f:
                nodefile = f

        # Process edges
        self.df = pl.read_parquet(edgefile)
        for c in EDGE_COLUMNS:
            assert c in self.df.columns
        self.temporal = True if 'timestamp' in self.df.columns else False
        if not self.temporal:
            self.df = self.df.with_columns(pl.lit(0).alias('timestamp'))
        self.datelist = self.df.select('timestamp').to_series().unique()

        # Process nodes
        if nodefile is not None:
            self.nodes = pl.read_parquet(nodefile)
        else:
            self.nodes = pl.concat([self.df.select(pl.col('timestamp'), pl.col('source').alias('nodes')), 
                                    self.df.select(pl.col('timestamp'), pl.col('dest').alias('nodes'))]).unique().sort(by=['timestamp','nodes'])
        for c in NODE_COLUMNS:
            assert c in self.nodes.columns

        self.edge_features = [x for x in self.df.columns if x not in ['timestamp', 'label']+EDGE_COLUMNS]
        self.node_features = [x for x in self.nodes.columns if x not in ['timestamp', 'label']+NODE_COLUMNS]

    def get_dates(self):
        return self.datelist.to_list()
    
    def get_edges(self):
        return self.df
    
    def get_nodes(self):
        return self.nodes

    def get_raphtory(self):
        g = rGraph()
        g.load_edges_from_pandas(
            df = self.df.to_pandas(),
            time = 'timestamp',
            src = 'source',
            dst = 'dest',
            properties = self.edge_features
        )
        g.load_nodes_from_pandas(
            df = self.nodes.to_pandas(),
            time = 'timestamp',
            id = 'nodes',
            properties = self.node_features
        )
        return g
    
    def get_edge_list(self, temp=True):
        if self.temporal and temp:
            edge_list = {}
            for d in tqdm(self.datelist):
                edges = self.df.filter(pl.col('timestamp')==d).select('source','dest').to_numpy()
                edge_list[d] = [tuple(x) for x in edges]
        else:
            edges = self.df.select('source','dest').unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
        return edge_list
    
    def get_networkx(self, temp=True):
        if self.temporal and temp:
            nx_graphs = {}
            for d in tqdm(self.datelist):
                edges = self.df.filter(pl.col('timestamp')==d).select('source','dest').to_numpy()
                edge_list = [tuple(x) for x in edges]
                nx_graphs[d] = nx.from_edgelist(edge_list)
        else:
            edges = self.df.select('source','dest').unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            nx_graphs = nx.from_edgelist(edge_list)
        return nx_graphs
    
    def get_edge_index(self, temp=True):
        if self.temporal and temp:
            edge_index = {}
            for d in tqdm(self.datelist):
                edges = self.df.filter(pl.col('timestamp')==d).select('source','dest').to_numpy()
                edge_list = [tuple(x) for x in edges]
                edge_index[d] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edges = self.df.select('source','dest').unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def get_tgeometric(self, temp=True):
        if self.temporal and temp:
            tg_graphs = {}
            for d in tqdm(self.datelist):
                edges = self.df.filter(pl.col('timestamp')==d).select('source','dest').to_numpy()
                edge_list = [tuple(x) for x in edges]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                tg = Data(edge_index=edge_index)
                nodes = self.nodes.filter(pl.col('timestamp')==d).select('nodes').to_numpy()
                tg.nodes = torch.from_numpy(nodes).int()
                features = self.nodes.filter(pl.col('timestamp')==d).select([c for c in self.node_features]).to_numpy()
                tg.x = torch.from_numpy(features).int()
                tg_graphs[d] = tg
        else:
            edges = self.df.select('source','dest').unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() 
            tg_graphs = Data(edge_index=edge_index)
            nodes = self.nodes.select('nodes').unique().to_numpy()
            tg_graphs.nodes = torch.Tensor(nodes).int()
        return tg_graphs
    
    def summary(self):
        pass

#
# From here on the code is obsolete and needs to be adjusted
#

def load_NFTS(divided_by_time=True, connected=True, days=178):
    DATAPATH = PATH+'ethereumSqlite/nfts.sqlite'

    ds = nfts.dataset.FromSQLite(DATAPATH)
    
    NFTS = pl.from_pandas(ds.load_dataframe("nfts"))
    transfers = pl.from_pandas(ds.load_dataframe("transfers"))
    current_owners = pl.from_pandas(ds.load_dataframe("current_owners"))
    
    
    
    transfers = transfers.with_columns(
            pl.from_epoch("timestamp", time_unit="s")
        ).sort('timestamp').select(['timestamp', 'from_address', 'to_address', 'transaction_value'])
    
    
    nodes=pl.concat([transfers['from_address'], transfers['to_address']])
    nodes=nodes.unique()
    new_nodes= {old: new for new, old in enumerate(nodes)}
    
    for c in transfers.columns[1:3]:
    
    
        transfers=transfers.with_columns(transfers[c].replace(new_nodes).alias(c))
        transfers=transfers.with_columns([transfers[c].cast(pl.Int32).alias(c)])
        
    transfers = transfers.with_columns(pl.from_epoch("timestamp", time_unit="d"))
    
    
    
    unique_dates = transfers['timestamp'].unique()
    days={old:new for new, old in enumerate(unique_dates)}
    
    transfers=transfers.with_columns(transfers['timestamp'].replace(days).cast(pl.Int32))
    
    
    numb_of_days=178
    
    edge_index=torch.tensor(transfers['from_address', 'to_address'].to_numpy(), dtype=torch.long)
    edge_index=edge_index.t().contiguous()
    edge_attr=torch.tensor(transfers['transaction_value'], dtype=torch.float)
    time=torch.tensor(transfers['timestamp'], dtype=torch.float)
    n=pl.concat([transfers['from_address'], transfers['to_address']])
    n=n.unique()
    n=torch.tensor(n)
    G=Data( edge_index=edge_index, edge_attr=edge_attr, time=time, nodes=n, num_nodes=len(n))
    if not divided_by_time:
        if connected:
            transform = LargestConnectedComponents()
            G_conn=transform(G)
            return G_conn
        else:
            return G
            
    else: 
    
        daily_transfers=[transfers.filter(pl.col('timestamp')==i) for i in range(numb_of_days)]
        dict_nodes_patches=[]
        
        daily_transfers_nodes_relabelled=[t.clone() for t in daily_transfers]
        df=daily_transfers_nodes_relabelled[0]
        n=pl.concat([df['from_address'], df['to_address']])
        n=n.unique()
        new_nodes_p= {old: new for new, old in enumerate(n)}
        dict_nodes_patches.append(new_nodes_p)
        for c in df.columns[1:3]:
            df=df.with_columns(df[c].replace(new_nodes_p).alias(c))
            df=df.with_columns([df[c].cast(pl.Int32).alias(c)])
        
        
        
        
        dict_nodes_patches=[]
        daily_transfers_nodes_relabelled=[]
        for df in daily_transfers:
            n=pl.concat([df['from_address'], df['to_address']])
            n=n.unique()
            new_nodes_p= {old: new for new, old in enumerate(n)}
            dict_nodes_patches.append(new_nodes_p)
            for c in df.columns[1:3]:
                df=df.with_columns(df[c].replace(new_nodes_p).alias(c))
                df=df.with_columns([df[c].cast(pl.Int32).alias(c)])
            daily_transfers_nodes_relabelled.append(df)
        
        
        daily_g=[]
        for df in daily_transfers_nodes_relabelled:
            ed=torch.tensor(df['from_address', 'to_address'].to_numpy(), dtype=torch.long)
            ed=ed.t().contiguous()
            ed_attr=torch.tensor(df['transaction_value'], dtype=torch.float)
            n=pl.concat([df['from_address'], df['to_address']])
            n=n.unique()
            n=torch.tensor(n)
            g=Data( edge_index=ed, edge_attr=ed_attr, nodes=n, num_nodes=len(n))
            daily_g.append(g)
        if connected:
            transform = LargestConnectedComponents()

            daily_g_conn=[transform(g) for g in daily_g]

            return daily_g_conn
        else:
            return daily_g

def summary(data):
    if not isinstance(data, type([])):
        data=[data]
    
    if isinstance(data[0], type(nx.Graph())):
        number_of_nodes=[]
        number_of_edges=[]
        avg_degree=[]
        for G in data:
            number_of_nodes.append(G.number_of_nodes())
            number_of_edges.append(G.number_of_edges())
            
            avg_degree.append(sum(dict(G.degree()).values())/G.number_of_nodes())
        avg_n=int(np.mean(number_of_nodes))
        avg_e=int(np.mean(number_of_edges))
        avg_d=int(np.mean(avg_degree))
            
        print('The dataset consists of {} graphs.\n Average number of nodes : {} \n Average number of edges : {} \n Average degree: {} '
                .format(len(data), avg_n, avg_e, avg_d))
    else:
        print('Converting into networkx graphs')
        data=[to_networkx(d, to_undirected=False) for d in tqdm(data)]
        summary(data)