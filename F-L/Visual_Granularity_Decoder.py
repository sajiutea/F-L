from torch import nn
import torch
from collections import OrderedDict

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class Query_Fusion(nn.Module):
    def __init__(self,
                 embed_dim,
                 # 自注意力层数
                 cmt_depth
                 ):
        super(Query_Fusion, self).__init__()
        self.embed_dim = embed_dim
        self.cmt_depth = cmt_depth        

        # 融合模块
        self.fusion_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.ln_fusion_q = LayerNorm(self.embed_dim)
        self.ln_fusion_kv = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        # 自注意力层
        self.fusion_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=self.cmt_depth,
                                                   heads=self.embed_dim // 64)
        # 初始化
        self.init_params()

    def init_params(self):
        scale = self.fusion_modal_transformer.width ** -0.5
        proj_std = scale * ((2 * self.fusion_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.fusion_modal_transformer.width) ** -0.5
        for block in self.fusion_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # init cross attn
        nn.init.normal_(self.fusion_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.fusion_attn.out_proj.weight, std=proj_std)

    def fusion_former(self, q, k, v):
        x = self.fusion_attn(
            self.ln_fusion_q(q),
            self.ln_fusion_kv(k),
            self.ln_fusion_kv(v),
            need_weights=False)[0]

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.fusion_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x

    def forward(self, q, k, v):
        # 扩展 query 以匹配 key 和 value 的批次大小
        batch_size = k.size(0)
        #q = q.expand(batch_size,-1, -1)
        feats = self.fusion_former(q, k, v)                
        return feats
