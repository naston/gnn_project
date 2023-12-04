from copy import deepcopy
from scipy.sparse import coo_matrix
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
import dgl

"""
This code is used to preprocess graph data before training.
"""

def remove_edges(G, edges):
    """
    Removes edges from graph. Used in creation of link prediction data splits.
    """
    G_new = deepcopy(G)
    G_new.remove_edges_from(edges)
    return G_new

def prepare_node_class(data):
    """
    Converts planetoid dataset to a usable format for dgl based models.
    """
    G = to_networkx(data, node_attrs=data.node_attrs(), to_undirected=data.is_undirected())
    train_g = dgl.from_networkx(G, node_attrs=list(G.nodes[0].keys()))
    return train_g

def create_train_test_split_edge(data):
    """
    Creates a training and a testing set for the link prediction task.
    """
    # Create a list of positive and negative edges
    u, v = data.edge_index.numpy()

    adj = coo_matrix((np.ones(data.num_edges), data.edge_index.numpy()))
    adj_neg = 1 - adj.todense() - np.eye(data.num_nodes)
    neg_u, neg_v = np.where(adj_neg != 0)

    # Create train/test edge split
    test_size = int(np.floor(data.num_edges * 0.1))
    eids = np.random.permutation(np.arange(data.num_edges)) # Create an array of 'edge IDs'

    train_pos_u, train_pos_v = data.edge_index[:, eids[test_size:]]
    test_pos_u, test_pos_v   = data.edge_index[:, eids[:test_size]]

    # Sample an equal amount of negative edges from  the graph, split into train/test
    neg_eids = np.random.choice(len(neg_u), data.num_edges)
    test_neg_u, test_neg_v = (
        neg_u[neg_eids[:test_size]],
        neg_v[neg_eids[:test_size]],
    )
    train_neg_u, train_neg_v = (
        neg_u[neg_eids[test_size:]],
        neg_v[neg_eids[test_size:]],
    )

    # Remove test edges from original graph
    G = to_networkx(data, node_attrs=data.node_attrs(), to_undirected=data.is_undirected())
    G_train = remove_edges(G, np.column_stack([test_pos_u, test_pos_v])) 

    train_g = dgl.from_networkx(G_train, node_attrs=list(G.nodes[0].keys()))

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=data.num_nodes)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=data.num_nodes)

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=data.num_nodes)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=data.num_nodes)

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g