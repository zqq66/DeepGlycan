import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

class DeepNorm(nn.Module):
    def __init__(self, normalized_shape, alpha, dropout_rate) -> None:
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, postx):
        return self.ln(x*self.alpha + self.dropout(postx))

class Relation(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 number_head: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float):
        super().__init__()
        self.number_head = number_head
        self.hidden_size = hidden_size

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
    
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

    def forward(self, node, node_mass):
        batch_size = node.size(0)
        node_num = node.size(1)
        q, k ,v = self.linear_q(node).view(batch_size, node_num, self.number_head, -1), self.linear_k(node).view(batch_size, node_num, self.number_head, -1),self.linear_v(node).view(batch_size, node_num,self.number_head, -1)
        node_mass = node_mass.unsqueeze(2)
        # print('q', torch.isnan(q).any(), 'k', torch.isnan(k).any(), 'node_mass', torch.isnan(node_mass).any())
        q, k = self.apply_rope(q, node_mass), self.apply_rope(k, node_mass)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]): 
            post_node = F.scaled_dot_product_attention(q,k,v,scale=1).transpose(1,2).flatten(2,3)
        post_node = self.output_layer(post_node)
        node = self.dn(node, post_node)
        return node

    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2,dim=-1)
        x0, x1 = x.chunk(2,dim=-1)
        return torch.concat([x0*dis_cos-x1*dis_sin,\
                             x1*dis_cos+x0*dis_sin], dim = -1)

class FFNGLU(nn.Module):
    def __init__(self, hidden_size: int, alpha: float, beta: float, dropout_rate: float):
        super().__init__()
        self.pre_ffn_gate = nn.Sequential(nn.Linear(hidden_size, 4*hidden_size, bias=False),
                                          nn.ReLU()
                                          )
        self.pre_ffn = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.post_ffn = nn.Linear(4*hidden_size, hidden_size, bias=False)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        nn.init.xavier_normal_(self.pre_ffn_gate[0].weight, gain=beta)
        nn.init.xavier_normal_(self.pre_ffn.weight, gain=beta)
        nn.init.xavier_normal_(self.post_ffn.weight, gain=beta)

    def forward(self, x):
        postx = self.post_ffn(self.pre_ffn_gate(x)*self.pre_ffn(x))
        x = self.dn(x, postx)
        return x

class DGEncoderLayer(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 number_head: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float):

        super().__init__()
        self.relation = Relation(hidden_size, number_head, alpha, beta, dropout_rate)
        self.ffn = FFNGLU(hidden_size, alpha, beta, dropout_rate)

    def forward(self, node, node_mass):
        #node = checkpoint(self.relation, node, node_mass, dist, predecessors, rel_mask)
        node = self.relation(node, node_mass)
        node = self.ffn(node)
        return node