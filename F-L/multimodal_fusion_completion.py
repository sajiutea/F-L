from torch import nn
import torch
from collections import OrderedDict

'''
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


# 一层 cross_attention 加上 (multi_head_attn + ffn) * 4 之后的 所有特征 Image: cross_image_feats、Text: cross_text_feats
class cross_attention(nn.Module):
    def __init__(self,
                 embed_dim,

                 # 自注意力层数
                 cmt_depth,

                 # 模态模式
                 modal_patten="Text"):
        super(cross_attention, self).__init__()
        self.embed_dim = embed_dim
        self.cmt_depth = cmt_depth
        self.modal_patten = modal_patten

        # 融合模块
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.ln_cross_q = LayerNorm(self.embed_dim)
        self.ln_cross_kv = LayerNorm(self.embed_dim)
        self.ln_cross_post = LayerNorm(self.embed_dim)

        # 自注意力层
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=self.cmt_depth,
                                                   heads=self.embed_dim // 64)
        # 初始化
        self.init_params()

    def init_params(self):
        scale = self.cross_modal_transformer.width ** -0.5
        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_cross_q(q),
            self.ln_cross_kv(k),
            self.ln_cross_kv(v),
            need_weights=False)[0]

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_cross_post(x)

        return x

    def forward(self, q, k, v):

        feats = self.cross_former(q, k, v)

        if self.modal_patten == "Text":
            cross_text_feats = feats
            return cross_text_feats
        else:
            cross_image_feat = feats[:, 0]
            return cross_image_feat


# 返回所有的融合特征: [mlm loss需要的特征 和 全局的视觉融合特征]
class Multimodal_Fusion_Completion(nn.Module):
    def __init__(self, embed_dim: int, image_mask: float, text_mask: bool, image_cmt_depth=1,text_cmt_depth=4):
        super().__init__()
        self.image_mask = image_mask
        self.text_mask = text_mask

        if self.image_mask:
            self.cross_image_attention = cross_attention(embed_dim, image_cmt_depth, "Image")

        if self.text_mask:
            self.cross_text_attention = cross_attention(embed_dim, text_cmt_depth, "Text")

    def forward(self, image_feats, text_feats, mlm_feats=None, mim_feats=None):
        ret = dict()
        # q,k,v
        if self.image_mask and mim_feats is not None:
            ret['cross_image_feat'] = self.cross_image_attention(mim_feats, text_feats, text_feats)
        # q,k,v
        if self.text_mask and mlm_feats is not None:
            ret['cross_text_feats'] = self.cross_text_attention(mlm_feats, image_feats, image_feats)

        return ret
'''

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


# 一层 cross_attention 加上 (multi_head_attn + ffn) * 4 之后的 所有特征 Image: cross_image_feats、Text: cross_text_feats
class cross_attention(nn.Module):
    def __init__(self,
                 embed_dim,
                 # 自注意力层数
                 cmt_depth
                 ):
        super(cross_attention, self).__init__()
        self.embed_dim = embed_dim
        self.cmt_depth = cmt_depth        

        # 融合模块
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.ln_cross_q = LayerNorm(self.embed_dim)
        self.ln_cross_kv = LayerNorm(self.embed_dim)
        self.ln_cross_post = LayerNorm(self.embed_dim)

        # 自注意力层
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=self.cmt_depth,
                                                   heads=self.embed_dim // 64)
        # 初始化
        self.init_params()

    def init_params(self):
        scale = self.cross_modal_transformer.width ** -0.5
        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_cross_q(q),
            self.ln_cross_kv(k),
            self.ln_cross_kv(v),
            need_weights=False)[0]
        
        # image_cross_feats.detch()
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_cross_post(x)

        return x

    def forward(self, q, k, v):
        feats = self.cross_former(q, k, v)        
        cross_text_feats = feats
        return cross_text_feats
        
class fusion_attention(nn.Module):
    def __init__(self,
                 embed_dim,
                 # 自注意力层数
                 cmt_depth
                 ):
        super(fusion_attention, self).__init__()
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
        feats = self.fusion_former(q, k, v)        
        fusion_image_feat = feats[:, 0]
        return fusion_image_feat
    
# 返回所有的融合特征: [mlm loss需要的特征 和 全局的视觉融合特征]
class Text_Fusion_Completion(nn.Module):
    def __init__(self, embed_dim: int, text_cmt_depth=4):
        super().__init__()      
    
        self.cross_text_attention = cross_attention(embed_dim, text_cmt_depth)

    def forward(self, image_feats, mlm_feats):
        
        cross_text_feats = self.cross_text_attention(mlm_feats, image_feats, image_feats)

        return cross_text_feats
    
#这里将cross变为fusion,是为了避开只要含有cross就学习率*5的设置
class Image_Fusion_Completion(nn.Module):
    def __init__(self, embed_dim: int, image_cmt_depth=4):
        super().__init__()      
    
        self.image_fusion_attention = fusion_attention(embed_dim, image_cmt_depth)

    def forward(self, text_feats, mim_feats):
        
        image_fusion_feat = self.image_fusion_attention(mim_feats, text_feats, text_feats)

        return image_fusion_feat