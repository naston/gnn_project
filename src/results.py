import pandas as pd
import numpy as np
import torch
import itertools
from torch_geometric.datasets import Planetoid, Actor
from torch_geometric.transforms import NormalizeFeatures
from .preprocess import prepare_node_class
from tqdm import tqdm

from .eval import compute_auc, compute_loss
from .preprocess import create_train_test_split_edge, prepare_node_class
from .models import GraphEVE, GraphSAGE, DotPredictor, MLPClassifier, GraphNSAGE

def run_edge_prediction(dataset,runs):
    data = Planetoid(root='data/Planetoid', name=dataset, transform=NormalizeFeatures())
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = create_train_test_split_edge(data[0])
    
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
                model = GraphEVE(train_g.ndata["x"].shape[1], 32)
            else:
                model = GraphSAGE(train_g.ndata["x"].shape[1], 32, agg=model_name)
            pred = DotPredictor()
            optimizer = torch.optim.Adam(
                itertools.chain(model.parameters(), pred.parameters()), lr=0.001
            )

            model_dict[model_name].append(train_edge_pred(500, model, pred, optimizer, 
                                                     train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g))
            
    return model_dict


def run_node_classification(dataset,runs):
    data = Planetoid(root='data/Planetoid', name=dataset, transform=NormalizeFeatures())
    #train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = create_train_test_split_edge(data[0])
    g = prepare_node_class(data[0])
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
                model = GraphEVE(g.ndata["x"].shape[1], data.num_classes, drop=0.5)
            else:
                model = GraphSAGE(g.ndata["x"].shape[1], data.num_classes, agg=model_name, drop=0.5)
            #pred = DotPredictor()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()

            model_dict[model_name].append(train_node_class(100, model, optimizer, criterion, g, data[0]))
            #g.ndata["feat"].shape[1], 16, dataset.num_classes
    return model_dict


def train_node_class(epochs, model, optimizer, criterion, g, dataset):
    accs = []
    for e in range(epochs):
        l, acc = train_node_class_epoch(model, optimizer, criterion, g, dataset)
        if (e+1)%5==0:
            accs.append(test_node_class(model,g))
    return np.max(accs)


def train_node_class_epoch(model, optimizer, criterion, g, dataset):
    model.train()
    optimizer.zero_grad()
    
    features = g.ndata["x"]
    #labels = dataset.ndata["label"]

    out = model(g, features)
    # Compute prediction
    #pred = out.argmax(1)
    #out = model(dataset.x, dataset.edge_index)
    loss = criterion(out[g.ndata['train_mask']], g.ndata['y'][g.ndata['train_mask']])
    #loss = F.cross_entropy(logits[train_mask], labels[train_mask])
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

def train_edge_pred(epochs, model, pred, optimizer, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g):
    aucs=[]
    for e in range(epochs):
        # forward
        
        h = model(train_g, train_g.ndata["x"])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # ----------- 5. check results ------------------------ #
        if (e+1) % 100 == 0:
                aucs.append(compute_auc(pos_score, neg_score))

    return np.max(aucs)


def joint_train(epochs, models, predictors, optimizers, criterion, data):
    res = []
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = data
    
    for e in range(500):
        for m in models.values():
            m.train()

        # Create embeddings and scores
        h = models['embed'](train_g, train_g.ndata['x'])
        h_link = models['link'](train_g, h)
        h_class = models['class'](train_g, h)
        class_logits = predictors['class'](h_class)
        pos_link_score = predictors['link'](train_pos_g, h_link)
        neg_link_score = predictors['link'](train_neg_g, h_link)
        

        class_loss = criterion(class_logits[train_g.ndata['train_mask']], train_g.ndata['y'][train_g.ndata['train_mask']])
        link_loss = compute_loss(pos_link_score, neg_link_score)

        embed_loss = class_loss.clone() + link_loss.clone()

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

    res.append((e, acc, auc))

    return res

def run_joint_train(dataset, runs, epochs=500):
    data = Planetoid(root='data/Planetoid', name=dataset, transform=NormalizeFeatures())

    train_data = create_train_test_split_edge(data[0])

    model_dict = {
        #'eve':[],
        #'mean':[],
        #'gcn':[],
        #'lstm':[],
        'pool':[],
    }
    for model_name in model_dict:
        for _ in tqdm(range(runs)):
            if model_name == 'eve':
                embed_model = GraphEVE(train_data[0].ndata["x"].shape[1], 32, drop=0.5)
                link_model = GraphNSAGE(32, 32, agg='mean', drop=0.5)
                class_model = GraphNSAGE(32, 32, agg='mean', drop=0.5)
            else:
                embed_model = GraphNSAGE(train_data[0].ndata["x"].shape[1], 32, agg=model_name, drop=0.5)
                link_model = GraphNSAGE(32, 32, agg=model_name, drop=0.5)
                class_model = GraphNSAGE(32, 32, agg=model_name, drop=0.5)

            link_pred = DotPredictor()
            class_pred = MLPClassifier(32, data.num_classes)
            
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

            res = joint_train(epochs, models, predictors, optimizers, criterion, train_data)
            model_dict[model_name].append(res)
    return model_dict



    