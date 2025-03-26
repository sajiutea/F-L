import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def compute_mlm(scores, labels):
    # ignore_index=0 表示在计算损失时忽略标签为 0 的位置，因为在 MLM 任务中，通常会使用 0 来表示未被掩码的单词。
    ce = nn.CrossEntropyLoss(ignore_index=0)
    loss = ce(scores, labels)   
    return loss

def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)    
    labels = labels.to(image_features.device)   

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)    
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits    
    logits_per_image = logit_scale * image_norm @ text_norm.t()   
    logits_per_text = logits_per_image.t()
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t)/2
 
    return loss

def Sinkhorn(K, u, v):
    max_iter = 1000
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for i in range(max_iter):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break

    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    return T
    
def compute_ot(image_features, text_features):
    eps = 0.1
    b = image_features.shape[0]   
    
    M = image_features.shape[1]
    N = text_features.shape[1]
    d = image_features.shape[-1]         

    image_features =  F.normalize(image_features, dim=-1) 
    
    text_features = F.normalize(text_features, dim=-1)
    
    
    sim = torch.einsum('bmd,bnd->bmn', image_features, text_features).contiguous()    #【32，196，127】    
    
    wdist = 1.0 - sim
    xx=torch.zeros(b, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
    yy=torch.zeros(b, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)

    with torch.no_grad():
        KK = torch.exp(-wdist / eps)
        T = Sinkhorn(KK,xx,yy)##【32，196，127】    
    if torch.isnan(T).any():
        return None
    
    # T*sim #[32,196,127]    
    sim_op = torch.sum(T * sim, dim=(1, 2))   #[32]
    
    return T

def get_sim(images, captions):     
    similarities = images.mm(captions.t())   
    return similarities

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        
        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = True

        self.false_hard = []

    def max_violation_on(self):
        self.max_violation = True

    def max_violation_off(self):
        self.max_violation = False

    def forward(self, im, s, scores=None):

        # compute image-sentence score matrix
        if scores is None:
            scores = get_sim(im, s)
        
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval, i->t
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval t->i
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals       
        mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)   

        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s, idx_s = cost_s.max(1)
            cost_im, idx_im = cost_im.max(0)

        loss = cost_s.sum() + cost_im.sum()

        return loss
        
def compute_cta(q_image, q_text, temperature,num_queries):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    bz = q_image.size(0)  
    
    #align          
    q_sim = torch.bmm(q_image, q_text.permute(0, 2, 1)) / temperature            
    q_num = q_sim.size(1)
    targets = torch.arange(q_num).type_as(q_text).long().repeat(bz)                

    q_sim_1 = rearrange(q_sim, "b n1 n2 -> (b n1) n2")
    loss_query_1 = torch.sum(F.cross_entropy(q_sim_1, targets, reduction="none")) / (bz * num_queries)  # mark

    q_sim_2 = rearrange(q_sim, "b n1 n2 -> (b n2) n1")
    loss_query_2 = torch.sum(F.cross_entropy(q_sim_2, targets, reduction="none")) / (bz * num_queries)

    cta_loss = (loss_query_1 + loss_query_2) / 2.
    return cta_loss

def compute_mvsc(image_feat_cls, mask_image_feat_cls):
    msm_temperature = 0.03

    image_feat_cls_n = image_feat_cls.norm(dim=-1).unsqueeze(-1)
    image_feat_cls = image_feat_cls / torch.max(image_feat_cls_n, 1e-8 * torch.ones_like(image_feat_cls_n))
    mask_image_feat_cls_n = mask_image_feat_cls.norm(dim=-1).unsqueeze(-1)
    mask_image_feat_cls = mask_image_feat_cls / torch.max(mask_image_feat_cls_n, 1e-8 * torch.ones_like(mask_image_feat_cls_n))

    # msm cls contrast
    sim_mt_img = torch.mm(mask_image_feat_cls, image_feat_cls.transpose(0, 1))
    loss_contrastive_func = NormSoftmaxLoss(msm_temperature)
    con_img_loss = loss_contrastive_func(sim_mt_img)

    return con_img_loss


def compute_mlsc(text_feat_cls, mask_text_feat_cls):
    msm_temperature = 0.03

    text_feat_cls_n = text_feat_cls.norm(dim=-1).unsqueeze(-1)
    text_feat_cls = text_feat_cls / torch.max(text_feat_cls_n, 1e-8 * torch.ones_like(text_feat_cls_n))
    mask_text_feat_cls_n = mask_text_feat_cls.norm(dim=-1).unsqueeze(-1)
    mask_text_feat_cls = mask_text_feat_cls / torch.max(mask_text_feat_cls_n, 1e-8 * torch.ones_like(mask_text_feat_cls_n))

    # msm cls contrast
    sim_mt_txt = torch.mm(mask_text_feat_cls, text_feat_cls.transpose(0, 1))
    loss_contrastive_func = NormSoftmaxLoss(msm_temperature)
    con_txt_loss = loss_contrastive_func(sim_mt_txt)

    return con_txt_loss

# 将 图片 -> transformer 格式输入
def patchify(imgs, patch_size=16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x

# 将 图片 -> transformer 格式输入
def res_patchify(imgs, patch_size=16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size*2
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x

# 根据原始的输入图像 和 重建后的图像计算损失
def compute_mim(imgs, pred, mask, norm_pix_loss):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """

    target = patchify(imgs)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

def compute_res(imgs, pred, mask, norm_pix_loss):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """

    target = res_patchify(imgs)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        """
            Assumes input x is similarity matrix of N x M \in [-1, 1],
            computed using the cosine similarity between normalised vectors
        """
        i_sim = x / self.temperature
        i_sim = i_sim - i_sim.max(dim=1, keepdim=True)[0]
        i_logsm = F.log_softmax(i_sim, dim=1)

        j_sim = x.t() / self.temperature
        j_sim = j_sim - j_sim.max(dim=1, keepdim=True)[0]
        j_logsm = F.log_softmax(j_sim, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

