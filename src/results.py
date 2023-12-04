import pandas as pd
import numpy as np
import torch
import itertools
from torch_geometric.datasets import Planetoid, Actor
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

from .eval import compute_auc, compute_loss
from .preprocess import create_train_test_split_edge, prepare_node_class
from .models import GraphEVE, GraphSAGE, DotPredictor, MLPPredictor, MLPClassifier, GraphNSAGE
from .parse_cora_text import get_raw_text_cora
from .parse_pubmed_text import get_raw_text_pubmed
from .parse_arxiv_text import get_raw_text_arxiv

from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset):
    if dataset=='Cora':
        data,_ = get_raw_text_cora(use_text=False)
    elif dataset=='PubMed':
        data,_ = get_raw_text_pubmed(use_text=False)
    elif dataset=='ogbn':
        data,_ = get_raw_text_arxiv(use_text=False)
    else:
        raise Exception('Unknown Embedding Dataset Combination')
    return data

def run_edge_prediction(dataset,runs, features='default', reg=0, pca=False):
    print('Running on:',device)
    
    if dataset=='ogbn':
        data = load_data(dataset)
    else:
        data = Planetoid(root='data/Planetoid', name=dataset, transform=NormalizeFeatures())[0]

    if features=='LM':
        
        data=load_data(dataset)
        feats = np.loadtxt(f'LM_embed_{dataset}.txt')
        if pca:
            feats = PCA().fit_transform(feats)
        data['x']=torch.tensor(feats)
    elif features=='TAPE':
        data=load_data(dataset)
        feats = np.loadtxt(f'TAPE_embed_{dataset}.txt')
        if pca:
            feats = PCA().fit_transform(feats)
            #feats = transform.fit_transform(feats)
        data['x']=torch.tensor(feats)
    elif features=='LM_pred':
        feats = np.loadtxt(f'LM_pred_{dataset}.txt')
        data['x']=torch.tensor(feats)
    elif features=='LM_full':
        feats = np.loadtxt(f'LM_embed_{dataset}.txt')
        preds = np.loadtxt(f'LM_pred_{dataset}.txt')
        feats = np.concatenate([feats,preds],axis=1)
        data['x']=torch.tensor(feats)
    elif features=='TAPE_full':
        lm = np.loadtxt(f'LM_embed_{dataset}.txt')
        tape = np.loadtxt(f'TAPE_embed_{dataset}.txt')
        preds = np.loadtxt(f'LM_pred_{dataset}.txt')
        feats = np.concatenate([lm,tape,preds],axis=1)
    print(data.x.shape)
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = create_train_test_split_edge(data)
    
    train_g=train_g.to(device)
    train_pos_g=train_pos_g.to(device)
    train_neg_g=train_neg_g.to(device)
    test_pos_g=test_pos_g.to(device)
    test_neg_g=test_neg_g.to(device)
    
    model_dict = {
        #'mean':[],
        #'gcn':[],
        #'lstm':[],
        'eve':[],
        #'pool':[],
    }
    for model_name in model_dict:
        for _ in tqdm(range(runs)):
            pred = DotPredictor()
            if model_name == 'eve':
                #continue
                model = GraphEVE(train_g.ndata["x"].shape[1], 32,drop=0.7)
                optimizer = torch.optim.Adam(
                    itertools.chain(model.parameters(), pred.parameters()), lr=0.001
                ) #0.0005
            else:
                model = GraphSAGE(train_g.ndata["x"].shape[1], 32, agg=model_name)
                optimizer = torch.optim.Adam(
                    itertools.chain(model.parameters(), pred.parameters()), lr=0.001
                )
                reg=0
            model=model.to(device)
            
            

            model_dict[model_name].append(train_edge_pred(500, model, pred, optimizer, 
                                                     train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g,reg))
            
    return model_dict


def run_node_classification(dataset,runs,features='default', reg=0, pca=False):
    print('Running on:',device)
    
    if dataset=='ogbn':
        data = load_data(dataset)
        n_classes=40   
    else:
        data = Planetoid(root='data/Planetoid', name=dataset, transform=NormalizeFeatures())
        n_classes = data.num_classes
        data = data[0]
    #train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = create_train_test_split_edge(data[0])
    if features=='LM':
        data=load_data(dataset)
        feats = np.loadtxt(f'LM_embed_{dataset}.txt')
        if pca:
            feats = PCA().fit_transform(feats)
        data['x']=torch.tensor(feats)
    elif features=='TAPE':
        data=load_data(dataset)
        feats = np.loadtxt(f'TAPE_embed_{dataset}.txt')
        if pca:
            feats = PCA().fit_transform(feats)
        data['x']=torch.tensor(feats)
    elif features=='LM_pred':
        feats = np.loadtxt(f'LM_pred_{dataset}.txt')
        data['x']=torch.tensor(feats)
    elif features=='LM_full':
        feats = np.loadtxt(f'LM_embed_{dataset}.txt')
        preds = np.loadtxt(f'LM_pred_{dataset}.txt')
        feats = np.concatenate([feats,preds],axis=1)
        data['x']=torch.tensor(feats)
    elif features=='TAPE_full':
        lm = np.loadtxt(f'LM_embed_{dataset}.txt')
        tape = np.loadtxt(f'TAPE_embed_{dataset}.txt')
        preds = np.loadtxt(f'LM_pred_{dataset}.txt')
        feats = np.concatenate([lm,tape,preds],axis=1)

    print(data['x'].shape)
    g = prepare_node_class(data)
    g = g.to(device)
    model_dict = {
        'mean':[],
        
        'gcn':[],
        #'lstm':[],
        'eve':[],
        'pool':[],
    }
    for model_name in model_dict:
        for _ in tqdm(range(runs)):
            if model_name == 'eve':
                #continue
                model = GraphEVE(g.ndata["x"].shape[1], n_classes, drop=0.5)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=0.01, weight_decay=5e-4
                )
                runs = 3
            else:
                model = GraphSAGE(g.ndata["x"].shape[1], n_classes, agg=model_name, drop=0.3)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=0.01, weight_decay=5e-4
                )
                reg=0
            #pred = DotPredictor()
            model=model.to(device)
            #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()

            model_dict[model_name].append(train_node_class(100, model, optimizer, criterion, g, reg))
            #g.ndata["feat"].shape[1], 16, dataset.num_classes
    return model_dict


def train_node_class(epochs, model, optimizer, criterion, g, reg=0):
    accs = []
    for e in range(epochs):
        l, acc = train_node_class_epoch(model, optimizer, criterion, g, reg)
        if (e+1)%5==0:
            accs.append(test_node_class(model,g).cpu())
    return np.max(accs)

def train_node_class_epoch(model, optimizer, criterion, g, reg=0):
    model.train()
    optimizer.zero_grad()
    
    features = g.ndata["x"]
    #labels = dataset.ndata["label"]

    out = model(g, features)
    # Compute prediction
    #pred = out.argmax(1)
    #out = model(dataset.x, dataset.edge_index)
    loss = criterion(out[g.ndata['train_mask']], g.ndata['y'][g.ndata['train_mask']])
    if reg>0:
        loss+= reg*torch.norm(list(model.conv1.pw_conv.parameters())[0].data - 0.5)
        loss+= reg*torch.norm(list(model.conv2.pw_conv.parameters())[0].data - 0.5)
    #loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=1)
    acc = pred[g.ndata['train_mask']] == g.ndata['y'][g.ndata['train_mask']]
    acc = acc.sum() / len(acc)
    return loss, acc

def test_node_class(model, g):
    model.eval()
    out = model(g ,g.ndata['x'])
    pred = out.argmax(dim=1)
    correct = pred[g.ndata['test_mask']] == g.ndata['y'][g.ndata['test_mask']]
    acc = correct.sum() / g.ndata['test_mask'].sum()

    return acc

def train_edge_pred(epochs, model, pred, optimizer, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, reg=0):
    aucs=[]
    for e in range(epochs):
        # forward
        
        h = model(train_g, train_g.ndata["x"])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        if reg>0:
            loss+= reg*torch.norm(list(model.conv1.pw_conv.parameters())[0].data - 0.5)
            loss+= reg*torch.norm(list(model.conv2.pw_conv.parameters())[0].data - 0.5)
            

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # ----------- 5. check results ------------------------ #
        if (e+1) % 100 == 0:
                test_pos_score = pred(test_pos_g, h)
                test_neg_score = pred(test_neg_g, h)
                aucs.append(compute_auc(test_pos_score, test_neg_score))

    return np.max(aucs)

def joint_train(epochs, models, predictors, optimizers, criterion, data, lamb):
    if lamb is None:
        lamb = lambda x, y: 0.5
    res = []
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = data
    
    for e in range(epochs):
        for m in models.values():
            m.train()

        # Create embeddings and scores
        h = models['embed'](train_g, train_g.ndata['x'])
        h_link = models['link'](train_g, h)
        h_class = models['class'](train_g, h)
        class_logits = predictors['class'](h_class)
        pos_link_score = predictors['link'](train_pos_g, h_link)
        neg_link_score = predictors['link'](train_neg_g, h_link)
        
        l = lamb(len(class_logits), len(pos_link_score) + len(neg_link_score))

        class_loss = criterion(class_logits[train_g.ndata['train_mask']], train_g.ndata['y'][train_g.ndata['train_mask']])
        link_loss = compute_loss(pos_link_score, neg_link_score)

        # Control parameter for the loss here
        embed_loss = (l * class_loss.clone() + (1 - l) * link_loss.clone()) * 2 # Added a * 2 to make the loss have the same weight relative to other losses as before

        optimizers['class'].zero_grad()
        optimizers['link'].zero_grad()
        optimizers['embed'].zero_grad()

        class_loss.backward(retain_graph=True)
        link_loss.backward(retain_graph=True)
        embed_loss.backward()

        optimizers['class'].step()
        optimizers['link'].step()
        optimizers['embed'].step()


    # Test Accuracy and AUC
    # ===================================================
    for m in models.values():
        m.eval()
    h = models['embed'](train_g, train_g.ndata['x'])
    h_link = models['link'](train_g, h)
    h_class = models['class'](train_g, h)
    class_logits = predictors['class'](h_class)

    pos_link_score = predictors['link'](test_pos_g, h_link)
    neg_link_score = predictors['link'](test_neg_g, h_link)
    auc = compute_auc(pos_link_score, neg_link_score)

    pred = class_logits.argmax(dim=1)
    correct = pred[train_g.ndata['test_mask']] == train_g.ndata['y'][train_g.ndata['test_mask']]
    acc = correct.sum() / len(correct)

    res.append((e, acc.cpu(), auc))

    return res

def run_joint_train(dataset, features, runs, epochs=500, lamb=None):
    #data = Planetoid(root='data/Planetoid', name=dataset, transform=NormalizeFeatures())

    data = Planetoid(root='data/Planetoid', name=dataset, transform=NormalizeFeatures())
    num_classes = data.num_classes
    data = data[0]
    if features=='LM':
        data=load_data(dataset)
        feats = np.loadtxt(f'LM_embed_{dataset}.txt')
        #if pca:
        #    feats = PCA().fit_transform(feats)
        data['x']=torch.tensor(feats)
    elif features=='TAPE':
        data=load_data(dataset)
        feats = np.loadtxt(f'TAPE_embed_{dataset}.txt')
        #if pca:
        #    feats = PCA().fit_transform(feats)
        #    feats = transform.fit_transform(feats)
        data['x']=torch.tensor(feats)

    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = create_train_test_split_edge(data)
    
    train_g=train_g.to(device)
    train_pos_g=train_pos_g.to(device)
    train_neg_g=train_neg_g.to(device)
    test_pos_g=test_pos_g.to(device)
    test_neg_g=test_neg_g.to(device)

    train_data = (train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g)
    model_dict = {
        'eve_node':[],
        'mean_node':[],
        'gcn_node':[],
        #'lstm':[],
        'pool_node':[],
        'eve_edge':[],
        'mean_edge':[],
        'gcn_edge':[],
        #'lstm':[],
        'pool_edge':[],
    }
    models = ['eve','mean','gcn','pool']

    for model_name in models:
        for _ in tqdm(range(runs)):
            if model_name == 'eve':
                embed_model = GraphEVE(train_data[0].ndata["x"].shape[1], 32, drop=0.5)
                link_model = GraphNSAGE(32, 32, agg='mean', drop=None)
                class_model = GraphNSAGE(32, 32, agg='mean', drop=None)
            else:
                embed_model = GraphNSAGE(train_data[0].ndata["x"].shape[1], 32, agg=model_name, drop=0.5)
                link_model = GraphNSAGE(32, 32, agg=model_name, drop=None)
                class_model = GraphNSAGE(32, 32, agg=model_name, drop=None)
            
            link_pred = DotPredictor()
            class_pred = MLPClassifier(32, num_classes)

            if torch.cuda.is_available():
                embed_model.cuda()
                link_model.cuda()
                class_model.cuda()
                class_pred.cuda()
                link_pred.cuda()
            
            optimizer_base = torch.optim.Adam(embed_model.parameters(), lr=0.01, weight_decay=5e-4)
            optimizer_class = torch.optim.Adam(itertools.chain(class_model.parameters(), class_pred.parameters()), lr=0.01, weight_decay=5e-4)
            optimizer_link =torch.optim.Adam(itertools.chain(link_model.parameters(), link_pred.parameters()), lr=0.01, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()

            models = {
                'embed': embed_model,
                'link': link_model,
                'class': class_model
            }

            predictors = {
                'link':link_pred,
                'class':class_pred
            }

            optimizers = {
                'embed':optimizer_base,
                'link':optimizer_link,
                'class':optimizer_class
            }

            res = joint_train(epochs, models, predictors, optimizers, criterion, train_data, lamb)
            res = np.array(res)
            res = np.max(res, axis=0)

            model_dict[model_name+'_node'].append(res[2])
            model_dict[model_name+'_edge'].append(res[1])
    return model_dict