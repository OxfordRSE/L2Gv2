import os
import logging
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

PATH = './data/snap-as/'

class DataLoader:
    
    def __init__(self, dataset='AS-733', path=PATH):
        
        if dataset=='AS-733':
            """Load autonomous systems graph data."""
            filelist = os.listdir(path)
            self.dates = sorted([s.replace('as','').replace('.txt','') for s in filelist])
            logger.info(f'Data available for {len(self.dates)} days between {self.dates[0]} and {self.dates[-1]}')

            self.num_nodes = 0
            self.graphs = {}
            for d in self.dates:
                filename = 'as'+d+'.txt'
                try:
                    edges = np.loadtxt(PATH+filename).astype('int64')
                except:
                    logger.critical("Could not open file", filename)
                self.num_nodes = max(edges.max(), self.num_nodes)
                self.graphs[d] = edges.transpose()
            
    def get_graph(self, date):
        return self.graphs[date]
    
    def get_all_graphs(self, format='list'):
        if format=='list':
            return [self.graphs[d] for d in self.dates]
        elif format=='dict':
            return {d: self.graphs[d] for d in self.dates}
    
        


