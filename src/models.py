from dgl.nn import SAGEConv
import dgl.function as fn
from dgl.utils import expand_as_pair
import torch
from torch import nn
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, agg='mean'):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
class GraphEVE(torch.nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphEVE, self).__init__()
        self.conv1 = EVEConv(in_feats, h_feats)
        self.conv2 = EVEConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
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
        

class EVEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(EVEConv, self).__init__()
        

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.dw_conv = nn.Conv2d(2, 1, 1)
        self.relu = nn.ReLU()
        self.fc_eve = nn.Linear(self._in_src_feats, out_feats, bias=False)

        self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)

        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)


        #self.reset_parameters()

    def reset_parameters(self):
        
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_max.weight, gain=gain)

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
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(msg_fn, fn.max('m', 'max'))
            graph.update_all(msg_fn, fn.min('m', 'min'))
            
            x_max = graph.dstdata['max']
            x_min = graph.dstdata['min']

            mxt = x_max.reshape(list(x_max.shape)+[1])
            mnt = x_min.reshape(list(x_min.shape)+[1])
            tt = torch.concat([mxt,mnt],dim=2)
            
            graph.dstdata['eve']=self.relu(self.dw_conv(tt.T).T)[:,:,0]

            h_eve = self.fc_eve(graph.dstdata['eve'])
            rst = self.fc_self(h_self) + h_eve

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