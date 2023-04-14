import torch
import pandas as pd
import numpy as np
import networkx as nx
is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000
Adj_file = 'PEMS04/PEMS04.csv'  #

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.Graph())  #

    return G

	
    return

def read_adj(Adj_file, num_nodes):
    nx_G = pd.read_csv(Adj_file)
    nx_G = nx_G.values
    adj_weight = torch.zeros(num_nodes, num_nodes)
    adj_rela = torch.zeros(num_nodes, num_nodes)
    for i, data in enumerate(nx_G):
        # print(i, data, data[0])
        a, b = int(data[0]), int(data[1])
        adj_rela[a][b] = 1
        adj_rela[b][a] = 1
        adj_weight[a][b] = data[-1]
    # print(adj_rela[158], adj_weight[158])
    print('return tensor adj:', adj_rela.shape)
    return adj_weight.float(), adj_rela.float()

if __name__ == '__main__':
    read_adj(Adj_file, 307)