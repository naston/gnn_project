from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd

"""
Code here allows us to gather abstract text in the same order as nodes in the ogbn-arxiv dataset.
This text is used to create LM embeddings.
"""

def get_raw_text_arxiv(use_text=False, seed=0):

    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv')#, transform=T.ToSparseTensor())#T.Compose([T.ToUndirected(),T.ToSparseTensor()]))#
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.y = data.y[:,0]

    #data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        './dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip',
        dtype={'node idx': 'int64','paper id': 'object'})

    raw_text = pd.read_csv('./data/ogbn_arxiv_orig/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])
    

    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = ab #'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text