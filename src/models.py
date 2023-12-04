from dgl.nn import SAGEConv
import dgl.function as fn
from dgl.utils import expand_as_pair
import torch
from torch import nn
import torch.nn.functional as F
import time

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, agg='mean', drop=0.0):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type=agg)
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type=agg)
        self.drop = drop

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)
        h = self.conv2(g, h)
        return h
    
class GraphNSAGE(torch.nn.Module):
    def __init__(self, *nfeats, agg='mean', drop=None):
        super(GraphNSAGE, self).__init__()
        self.convs = nn.ModuleList([])
        for i in range(len(nfeats)-1):
            self.convs.append(SAGEConv(nfeats[i], nfeats[i+1], aggregator_type=agg))
        self.drop = drop

    def forward(self, g, h):
        for i, c in enumerate(self.convs):
            h = c(g, h)
            # if self.drop is not None:
            #     h = F.relu(h)
            #     h = F.dropout(h, p=self.drop, training=self.training)
        return h

class GraphEVE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, drop=0.0):
        super(GraphEVE, self).__init__()
        self.conv1 = EVEConv(in_feats, h_feats)
        self.conv2 = EVEConv(h_feats, h_feats)
        self.drop = drop

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)
        h = self.conv2(g, h)
        return h
    
class GraphIEVE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, drop=0.0):
        super(GraphIEVE, self).__init__()
        self.conv1 = IEVEConv(in_feats, h_feats)
        self.conv2 = IEVEConv(h_feats, h_feats)
        self.drop = drop

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)
        h = self.conv2(g, h)
        return h

class DotPredictor(torch.nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]
        
class MLPPredictor(torch.nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_feats, 32)
        self.linear2 = torch.nn.Linear(32, 1)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_add_v("h", "h", "score"))

            score = g.edata['score']

            score = self.linear1(score)
            score = F.relu(score)
            score = self.linear2(score)
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return score.squeeze()

class MLPClassifier(torch.nn.Module):
    def __init__(self, in_feats, num_classes):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_feats, 32)
        self.linear2 = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # x = F.softmax(x) - Not needed when performing loss
        return x

class EVEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.3,
                 bias=True,
                 norm=None,
                 activation=None):
        super(EVEConv, self).__init__()
        

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.drop = nn.Dropout(0.2)

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.pw_conv = nn.Conv2d(2, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc_eve = nn.Linear(self._in_src_feats, out_feats, bias=False)

        self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)

        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)


        self.reset_parameters()

    def reset_parameters(self):
        
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_eve.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            
            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['max'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Message Passing
            #print('Timing:')
            #t0 = time.time()
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(msg_fn, fn.max('m', 'max'))
            graph.update_all(msg_fn, fn.min('m', 'min'))
            
            #t1 = time.time()
            #print('\tUpdate:',t1-t0)

            x_max = graph.dstdata['max']
            x_min = graph.dstdata['min']

            #t2 = time.time()
            #print('\tPull:',t2-t1)

            mxt = x_max.reshape([1]+list(x_max.shape))
            mnt = x_min.reshape([1]+list(x_min.shape))
            tt = torch.concat([mxt,mnt],dim=0)

            #t3 = time.time()
            #print('\tReshape:',t3-t2)
            
            eve=torch.squeeze(self.pw_conv(tt))

            #print(eve.shape)

            #t4 = time.time()
            #print('\tPWC:',t4-t3)

            h_eve = self.fc_eve(self.drop(eve))
            rst = self.fc_self(h_self) + h_eve
            #t5 = time.time()
            #print('\tFinal:',t5-t4)
            #print('Time:',t5-t0)

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            #print(rst.shape)
            return rst
        
class IEVEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.3,
                 bias=True,
                 norm=None,
                 activation=None):
        super(IEVEConv, self).__init__()
        

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.drop = nn.Dropout(0.2)

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        #self.pw_conv = nn.Conv2d(2, 1, 1)
        self.relu = nn.ReLU()
        self.fc_eve = nn.Linear(self._in_src_feats, out_feats, bias=False)

        

        self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)

        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)


        self.reset_parameters()

    def reset_parameters(self):
        
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_eve.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            
            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['max'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Message Passing
            #print('Timing:')
            #t0 = time.time()
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(msg_fn, fn.max('m', 'max'))
            graph.update_all(msg_fn, fn.min('m', 'min'))
            
            #t1 = time.time()
            #print('\tUpdate:',t1-t0)

            x_max = graph.dstdata['max']
            x_min = graph.dstdata['min']

            #t2 = time.time()
            #print('\tPull:',t2-t1)

            mxt = x_max.reshape([1]+list(x_max.shape))
            mnt = x_min.reshape([1]+list(x_min.shape))
            
            tt = torch.concat([mxt,mnt],dim=0)

            #t3 = time.time()
            #print('\tReshape:',t3-t2)
            
            #eve=torch.squeeze(self.pw_conv(tt))

            eve = torch.squeeze(F.interpolate(tt.permute(*torch.arange(tt.ndim - 1, -1, -1)), size=1, mode='linear')).T

            #t4 = time.time()
            #print('\tPWC:',t4-t3)

            h_eve = self.fc_eve(self.drop(eve))
            rst = self.fc_self(h_self) + h_eve
            #t5 = time.time()
            #print('\tFinal:',t5-t4)
            #print('Time:',t5-t0)

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            #print(rst.shape)
            return rst