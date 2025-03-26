import torch
from torch import nn
from transformers import SwinConfig, SwinModel
import torch
from lateral_vit import load_vit



class LateralImageEncoder(nn.Module):
    def __init__(
        self,
        lateral: str
    ):
        super().__init__()
        self.lateral = lateral
        if lateral=='swin':
            self.swin_encoder = load_swin()
            self.out_dim = self.swin_encoder.config.hidden_size
            self.adapter = nn.Sequential(nn.Linear(self.out_dim,self.out_dim), nn.Tanh(),
                 nn.Linear(self.out_dim, 512), nn.Softmax(dim=1))
        else:
            self.vit_encoder = load_vit()
    def forward(self, image):
        if self.lateral=="swin":
            #冻结swin的参数
            for param in self.swin_encoder.parameters():
                param.requires_grad = False
            output = self.swin_encoder(pixel_values=image)
            img_feats = output["last_hidden_state"]
            #print(img_feats.shape)#torch.Size([1, 49, 768])#torch.Size([3, 49, 768])
            aggragate_feats = self.adapter(img_feats)
            return aggragate_feats  # (batch, seq_len, hidden_size)
        else:
            img_feats = self.vit_encoder(image) #[batch,197,512]
            return img_feats  # (batch, seq_len, hidden_size)

def attention_fn(query, context, temp1):
    """
    query: batch x queryL x ndf
    context: batch x sourceL x ndf
    temp1: scalar to adjust the softmax temperature
    """
    # query和context已经具有正确的维度，可以直接计算

    # 注意力机制
    # 首先，由于bmm需要后两维进行矩阵乘法，需要将context转置
    context_transposed = torch.transpose(context, 1, 2)  # batch x ndf x sourceL
    
    # 使用bmm计算未规范化的注意力得分
    attn = torch.bmm(query, context_transposed)  # batch x queryL x sourceL

    # 应用温度调节因子
    attn = attn / temp1

    # 应用Softmax归一化处理, 注意力分布沿sourceL维度
    attn = nn.Softmax(dim=-1)(attn)

    # 使用注意力得分加权上下文
    # 这里需要将context用原始的维度 batch x sourceL x ndf
    weightedContext = torch.bmm(attn, context)  # batch x queryL x ndf

    return weightedContext, attn

def get_aggragate_img_feats(img_features_l, img_features_f, ind_lateral, words_emb):

    batch_size = img_features_f.shape[0]#[32,197,512]
    img_features_f = img_features_f.type(words_emb.dtype)   
    img_features_l = img_features_l.type(words_emb.dtype)    
    
    if torch.all(ind_lateral):#无侧面图像
        img_feats_lateral_all = 1e-8 * torch.ones((batch_size,49,512), dtype=words_emb.dtype).cuda()
    else:#有侧面图像
        img_feats_lateral_all = torch.zeros((batch_size,49,512), dtype=words_emb.dtype).cuda()
        img_feats_lateral_all[~ind_lateral] = img_features_l
        img_feats_lateral_all[ind_lateral] = 1e-8 * torch.ones((49,512),dtype=words_emb.dtype).cuda()
      
    # 使用掩码处理
    valid_mask = ~ind_lateral   
    
    img_features = torch.cat((img_features_f, img_feats_lateral_all), dim=1)#[32,245,512]    
    #weiContext, attn = attention_fn(words_emb, img_features, 4.0)#[32,97,512]      
    return img_features,img_feats_lateral_all,valid_mask #[32,392,512]

def load_swin():
    configuration = SwinConfig()
    model = SwinModel(configuration)

    medical_pretrained = '/root/autodl-fs/swint_mcc.tar'
    checkpoint = torch.load(medical_pretrained,map_location="cpu")
    checkpoint_model = checkpoint['model']

    img_state_dict = {}
    for k, v in checkpoint_model.items():
        if k.startswith('image_encoder.image_encoder.'):
            img_state_dict[k[len('image_encoder.image_encoder.'):]] = v

    msg = model.load_state_dict(img_state_dict, strict=True)
    print(msg)
    return model