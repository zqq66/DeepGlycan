import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

class DeepNorm(nn.Module):
    def __init__(self, normalized_shape, alpha, dropout_rate) -> None:
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, postx):
        # print('x, postx', x.shape, postx.shape)
        return self.ln(x * self.alpha + self.dropout(postx))


class MaskedMultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_head: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()
        head_size = hidden_size // n_head
        assert hidden_size % 8 == 0
        assert head_size % 8 == 0 and head_size <= 128
        assert hidden_size % n_head == 0

        self.n_head = n_head
        self.head_size = head_size
        self.hidden_size = hidden_size

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

    def forward(self, tgt, glycan_mass, glycan_query_mass):
        batch_size = tgt.size(0)
        q = self.linear_q(tgt).view(batch_size, -1, self.n_head, self.head_size)
        k = self.linear_k(tgt).view(batch_size, -1, self.n_head, self.head_size)
        v = self.linear_v(tgt).view(batch_size, -1, self.n_head, self.head_size)
        q = self.apply_rope(q, glycan_query_mass)
        k = self.apply_rope(k, glycan_mass)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]): 
            postx = F.scaled_dot_product_attention(q,k,v,scale=1,is_causal=True).transpose(1,2).flatten(2,3)


        postx = self.output_layer(postx)
        tgt = self.dn(tgt, postx)
        return tgt


    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2, dim=-1)
        # print('dis_sin, dis_cos', dis_sin.shape, dis_cos.shape)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        # print('x0, x1', x0.shape, x1.shape)
        return torch.concat([x0 * dis_cos - x1 * dis_sin, \
                             x1 * dis_cos + x0 * dis_sin], dim=-1)


class MultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_head_per_mass: int,
                 key_size: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()
        assert hidden_size % 8 == 0
        assert key_size % 8 == 0 and key_size <= 128

        self.key_size = key_size
        self.n_head_projection = n_head_per_mass
        self.linear_q = nn.Linear(hidden_size, self.n_head_projection * key_size)
        self.linear_k = nn.Linear(hidden_size, self.n_head_projection * key_size)
        self.linear_v = nn.Linear(hidden_size, self.n_head_projection * key_size)
        self.output_layer = nn.Linear(self.n_head_projection * key_size, hidden_size)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

    def forward(self, tgt, mem, glycan_mass):
        """_summary_

        Args:
            tgt (_type_): _description_
            mem_key_padding_mask (_type_): _description_
            pep_mass (_type_): _description_
            mem (_type_, optional): _description_. Defaults to None.
            peaks_moverz (_type_, optional): _description_. Defaults to None.
            past_mem (_type_, optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        batch_size = tgt.size(0)
        q = self.linear_q(tgt).view(batch_size, -1, self.n_head_projection, self.key_size)
        k = self.linear_k(mem).view(batch_size, -1, self.n_head_projection, self.key_size)
        v = self.linear_v(mem).view(batch_size, -1, self.n_head_projection, self.key_size)
        # print('forward node_mass', node_mass.shape)
        q = self.apply_rope(q, glycan_mass)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]): 
            postx = F.scaled_dot_product_attention(q,k,v,scale=1).transpose(1,2).flatten(2,3)

        postx = self.output_layer(postx)
        # print('tgt', tgt.shape, v.shape)

        tgt = self.dn(tgt, postx)
        return tgt


    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2, dim=-1)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0 * dis_cos - x1 * dis_sin, \
                             x1 * dis_cos + x0 * dis_sin], dim=-1)


class FFN(nn.Module):
    def __init__(self, hidden_size: int, alpha: float, beta: float, dropout_rate: float):
        super().__init__()

        self.ffn = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4 * hidden_size, hidden_size, bias=False))
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)

        nn.init.xavier_normal_(self.ffn[0].weight, gain=beta)
        nn.init.xavier_normal_(self.ffn[-1].weight, gain=beta)

    def forward(self, x):
        postx = self.ffn(x)
        x = self.dn(x, postx)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, n_head, alpha, beta, dropout_rate):
        super().__init__()
        self.mmha = MaskedMultiHeadAttn(hidden_size=hidden_size,
                                        n_head=n_head,
                                        alpha=alpha,
                                        beta=beta,
                                        dropout_rate=dropout_rate)

        self.mha = MultiHeadAttn(hidden_size=hidden_size,
                                 n_head_per_mass=n_head,
                                 key_size=hidden_size // n_head,
                                 alpha=alpha,
                                 beta=beta,
                                 dropout_rate=dropout_rate)

        self.ffn = FFN(hidden_size=hidden_size,
                       alpha=alpha,
                       beta=beta,
                       dropout_rate=dropout_rate)

    def forward(self, tgt,mem, glycan_mass, glycan_query_mass,glycan_crossattn_mass):
        tgt = self.mmha(tgt, glycan_mass, glycan_query_mass)
        # print('forward', tgt[0])
        tgt = self.mha(tgt,
                       mem=mem,
                       glycan_mass=glycan_crossattn_mass,)
        tgt = self.ffn(tgt)

        return tgt

