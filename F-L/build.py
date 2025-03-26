from load_model import load_model_checkpoint
import torch
import torch.nn as nn
from collections import OrderedDict
from multimodal_fusion_completion import Text_Fusion_Completion,Transformer
import new_objectives
from new_objectives import ContrastiveLoss
from transformers import AutoTokenizer
import torch.nn.functional as F
import numpy as np
from Visual_Granularity_Decoder import Query_Fusion
from itm_fusion import Match_Fusion
from einops import rearrange
from lateral_swintiny import LateralImageEncoder,get_aggragate_img_feats
from token_selection import CrossSparseAggrNet_v2
from relation import RN
from torch import distributed as dist

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# 基于Pytorch框架,整个模型由两部分组成:
#   1. 基于Clip架构的特征提取器(Text-encoder 和 image-encoder)
#   2. 多模态融合模块
class MMR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 设置训练目标
        self._set_task()
        self.base_model = load_model_checkpoint(args)
        self.logit_scale =torch.ones([]) * (1 / self.args.temperature)
        self.embed_dim = 512        
        self.cls_id = 101 #101
        self.sparse_ratio = 0.8
        self.aggr_ratio = 0.7
        self.num_queries = self.args.num_queries
        self.image_mask = self.args.image_mask
        self.text_mask = self.args.text_mask
        
        self.text_cmt_depth = self.args.text_cmt_depth

        self.query_cmt_depth = self.args.query_cmt_depth
        self.text_cross_former = Text_Fusion_Completion(self.embed_dim,self.text_cmt_depth)
        
        self.bert_model_name = '/root/autodl-fs/Bioclinical_BERT'
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name,trust_remote_code=True)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}        
        
        self.query = nn.Parameter(torch.randn(1,self.num_queries, self.embed_dim),requires_grad=False)
        self.query = nn.init.xavier_uniform_(self.query)#Xavier/Glorot 初始化        
        
        self.granularity_decoder = Query_Fusion(self.embed_dim,self.query_cmt_depth)
        self.match_cmt_depth = self.args.match_cmt_depth
        #self.itm_fusion = Match_Fusion(self.embed_dim,self.match_cmt_depth)
        #self.itm_head = nn.Linear(self.embed_dim, 2)     
        #冻结的swintransformer
        self.lateral_encoder = LateralImageEncoder(self.args.lateral)
        self.Selection = CrossSparseAggrNet_v2(self.embed_dim,self.sparse_ratio,self.aggr_ratio)
        
        self.prototype_layer = nn.Linear(self.embed_dim, 500, bias=False)
        self.epsilon = 0.07 #0.07
        self.sinkhorn_iterations = 3
        self.get_assignments = self.distributed_sinkhorn
        self.proto_temperature = 0.1
        self.gpus=3
        self.fl_fusion_layer = nn.Linear(2*self.embed_dim, 512)
        if 'mlm' in args.loss_names:
            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            cross_modal_transformer_width = self.embed_dim
            cross_modal_transformer_layers = self.text_cmt_depth
            scale = cross_modal_transformer_width ** -0.5
            proj_std = scale * ((2 * cross_modal_transformer_layers) ** -0.5)
            fc_std = (2 * cross_modal_transformer_width) ** -0.5

            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        
    def distributed_sinkhorn(self, Q, nmb_iters):#采用分布式训练
        with torch.no_grad():#在此过程中不会计算梯度
            sum_Q = torch.sum(Q)#计算 Q 的总和，即分布 Q 中所有元素的和。
            dist.all_reduce(sum_Q)#使用分布式训练（例如，使用多个 GPU）时，将所有设备上的 sum_Q 的值相加，以确保所有设备上的值都一致。
            Q /= sum_Q
            
            u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(
                non_blocking=True) / (self.gpus * Q.shape[1])          
            curr_sum = torch.sum(Q, dim=1)
            
            dist.all_reduce(curr_sum)
            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()           
    
    # _set_task 方法：设置当前任务，根据传入的 loss_names。
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')       
    
    def forward(self, batch):

        caption_ids = batch['caption_ids']
        
        prediction = self.base_model(batch)
        image_feat = prediction['unmasked_image_feat']
        text_feat = prediction['unmasked_text_feat']        
        '''
        ret:
        if itc:
            unmasked_i_feat
            unmasked_t_feat
        if mvsc:
            masked_i_feat
        if mlm:
            mlm_cls_feats
        if mlsc:
            masked_t_feat
        '''                           
        if 'itm' in self.current_task:
            bs = image_feat.size(0)
            image_embeds = prediction['unmasked_image_feats']            
            text_embeds = prediction['unmasked_text_feats']
            
            output_pos = self.itm_fusion(text_embeds,image_embeds,image_embeds)
            with torch.no_grad():                
                image_norm = image_feat / image_feat.norm(dim=-1, keepdim=True)    
                text_norm = text_feat / text_feat.norm(dim=-1, keepdim=True)

                # cosine similarity as logits
                weights_i2t = F.softmax(self.logit_scale * image_norm @ text_norm.t(),dim=1)
                weights_t2i = F.softmax(self.logit_scale * text_norm @ image_norm.t(),dim=1)           
                
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)

            # select a negative image for each text
            image_embeds_neg = []        
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()                                    
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text for each image
            text_embeds_neg = []              
            for b in range(bs):                
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()                                       
                text_embeds_neg.append(text_embeds[neg_idx])   
                
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)   
            
            text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)           
            image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)            

            output_neg = self.itm_fusion(text_embeds_all,image_embeds_all,image_embeds_all)

            vl_embeddings = torch.cat([output_pos[:, 0, :], output_neg[:, 0, :]], dim=0)
            vl_output = self.itm_head(vl_embeddings)  # self.itm_head = nn.Linear(text_width, 2)
            itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)
            itm_loss = F.cross_entropy(vl_output, itm_labels)*self.args.itm_loss_weight                        
               
        if 'mlm' in self.current_task:
            mlm_feats = prediction['masked_text_feats']  # [1,77,768]
            img_feats = prediction['unmasked_image_feats']
            
            img_cls = prediction['unmasked_image_feats'][:,0,:]
            image_feats_wocls = prediction['unmasked_image_feats'][:,1:,:]
            text_feats = prediction['unmasked_text_feats']
            
            l_images = batch["l_images"]   
            ind_lateral = ~batch['is_lateral']
            
            # feed forward to the feature extractor only cases with lateral images
            imgs_l_continue = l_images[~ind_lateral]#有侧面图像：torch.Size([1, 3, 224, 224]) 无侧面图像：torch.Size([0, 3, 224, 224])            
            num_lateral = imgs_l_continue.size(0)
            
            if imgs_l_continue.numel() == 0:
                #print("Warning: imgs_l_continue is empty.")
                img_emb_l_l = torch.zeros((l_images.size(0),49,512))
            else:
                img_emb_l_l = self.lateral_encoder(imgs_l_continue)
            
            f_l_img_feats,l_img_feats,valid_mask = get_aggragate_img_feats(img_emb_l_l, image_feats_wocls, ind_lateral, text_feats) # [32,245,512]  24,392,512]            
             
            select_tokens = self.Selection(f_l_img_feats,text_feats) #[32,118,512] [24,79,512]
            
            #img_cls = img_cls.unsqueeze(1)
            # 在第二维度上拼接
            #final_img_feats = torch.cat((img_cls, select_tokens), dim=1)
            
            # 5 Transformer 操作
            cross_text_feats = self.text_cross_former(select_tokens, mlm_feats)  # q,k,v
            # 并使用 mlm 头对结果进行分类
            x = self.mlm_head(cross_text_feats)  # [batch_size, text_len, num_colors]
            scores = x.float().reshape(-1, self.args.vocab_size)
            # print(scores.shape)
            # .reshape(-1) 操作将张量变为一维张量
            mlm_labels = batch['mlm_labels'].reshape(-1)
            # print(mlm_labels.shape)
            mlm_loss= new_objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight
            # 计算准确率
            # 这行代码计算了在每一行中具有最大值的索引，然后将这些索引存储在 pred 变量中
            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            mlm_acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()         
        if 'cpa' in self.current_task:
            # normalize prototype layer对齐原型层
            with torch.no_grad():
                w = self.prototype_layer.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                self.prototype_layer.weight.copy_(w)
            limg_feat = torch.mean(l_img_feats, dim=1)
            # Compute assign code of images
            fimg_proto_out = self.prototype_layer(image_feat)
            limg_proto_out = self.prototype_layer(limg_feat)
            report_proto_out = self.prototype_layer(text_feat)
            with torch.no_grad():                
                report_code = torch.exp(
                    report_proto_out / self.epsilon).t()
                report_code = self.get_assignments(
                    report_code, self.sinkhorn_iterations)       # bz, 500           
            fimg_proto_prob = F.softmax(
                fimg_proto_out / self.proto_temperature, dim=1)
            limg_proto_prob = F.softmax(          
                limg_proto_out / self.proto_temperature, dim=1)            

            loss_f2t_proto = - \
                torch.mean(torch.sum(report_code * torch.log(fimg_proto_prob), dim=1))
            loss_l2t_proto = - \
                torch.mean(torch.sum(report_code * torch.log(limg_proto_prob), dim=1))
           
            cpa_loss = (loss_f2t_proto + loss_l2t_proto) / 2.
           
        if 'itc' in self.current_task:
            limg_feat = torch.mean(l_img_feats, dim=1)
            fl_feat = torch.cat((image_feat, limg_feat), dim=1)
            fl_fusion_feat = self.fl_fusion_layer(fl_feat)            
            ftc_loss = new_objectives.compute_itc(image_feat, text_feat, self.logit_scale)  
            ltc_loss = new_objectives.compute_itc(limg_feat, text_feat, self.logit_scale)             
            fltc_loss = new_objectives.compute_itc(fl_fusion_feat, text_feat, self.logit_scale)
            itc_loss= (ftc_loss+ltc_loss)/2.+fltc_loss
        
        if 'cta' in self.current_task:
            local_image_feats = prediction['unmasked_image_feats'][:, 1:,:]#torch.Size([24, 196, 512])
            local_text_feats = prediction['unmasked_text_feats'][:, 1:,:]#torch.Size([24, 127, 512]) 
            bz = local_image_feats.size(0)            
            
            #图像、query  
            q_limage = self.granularity_decoder(local_text_feats,l_img_feats,l_img_feats)            
            q_limage = F.normalize(q_limage, dim=-1)
            #文本、query
            q_fimage = self.granularity_decoder(local_text_feats,local_image_feats,local_image_feats)
            q_fimage = F.normalize(q_fimage, dim=-1)
            #cta_loss = new_objectives.compute_cta(q_image, q_text, self.args.local_temperature,self.num_queries) * self.args.cta_loss_weight              
            
             #align          
            q_sim = torch.bmm(q_fimage, q_limage.permute(0, 2, 1)) / self.args.local_temperature               
            
            q_num = q_sim.size(1)
            targets = torch.arange(q_num).type_as(q_fimage).long().repeat(bz)                
            
            q_sim_1 = rearrange(q_sim, "b n1 n2 -> (b n1) n2")
            
            loss_query_1 = torch.sum(F.cross_entropy(q_sim_1, targets, reduction="none")) / (bz * q_num) #(bz * self.num_queries)  
            
            q_sim_2 = rearrange(q_sim, "b n1 n2 -> (b n2) n1")
            loss_query_2 = torch.sum(F.cross_entropy(q_sim_2, targets, reduction="none")) / (bz * q_num) #(bz * self.num_queries)
            cta_loss = (loss_query_1 + loss_query_2) / 2. *self.args.cta_loss_weight 
        
        return itc_loss, mlm_loss, mlm_acc, cta_loss,cpa_loss

# 构建模型，构建并返回 IRRA 类的实例，并将模型权重转换为半精度浮点数（fp16）。
def build_model(args):
    model = MMR(args)
    return model.eval()

