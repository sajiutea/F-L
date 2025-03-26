import torch
import numpy as np
import random
from Mask_VIT import Masked_Vit
from typing import Tuple, Union
from transformers import AutoTokenizer, AutoModel
from torch import nn


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


class Masked_Clip(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            # vision image_size: 224 × 224
            image_resolution: int,
            vision_layers: Union[Tuple[int, int, int, int], int],
            vision_width: int,
            vision_patch_size: int,

            # text
            text_encoder_width : float,

            # mask
            image_mask : float,
            text_mask = False
    ):
        super().__init__()

        self.image_mask = image_mask
        self.text_mask = text_mask

        # image_encoder
        self.visual = Masked_Vit(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size
        )

        # text_encoder
        self.cls_id = 101 #101
        bert_name = '/root/autodl-fs/Bioclinical_BERT'
        self.text_encoder = AutoModel.from_pretrained(bert_name, trust_remote_code=True)
        self.text_encoder_width = text_encoder_width
        self.text_projection = nn.Parameter(torch.empty(self.text_encoder_width, embed_dim))
        self.init_parameters()
    
    def init_parameters(self):
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.text_encoder_width ** -0.5)
    
    def encode_image(self, image, mask_ratio):
        ret = self.visual(image, mask_ratio)
        return ret

    # 没有归一化的特征(bert貌似有ln)
    def encode_text(self, text):
        """
            text_output:
                text_feat
                text_feats
        """
        embeddings = self.text_encoder(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        text_features = embeddings['last_hidden_state']

        # 投影
        text_features = text_features @ self.text_projection       

        # 提取全局特征(?) [1, 77, 768] -> [:, 0]
        last_token_index = torch.nonzero((text['input_ids'] == self.cls_id))        
        text_feat = text_features[torch.arange(text_features.shape[0]), last_token_index[:, 1]]

        text_output = dict()
        text_output['text_feat'] = text_feat
        text_output['text_feats'] = text_features

        return text_output

    def forward(self, batch):
        """
            batch:
                # image
                images

                # text
                texts
                caption_ids
                attention_mask

            image_mask: image_mask_ratio
            text_mask: is_mask_text, set text_mask_ratio in other files
        """

        images = batch["images"]  # [1,3,224,224];        
        texts = {'input_ids': batch['caption_ids'],
                'attention_mask': batch['attention_mask']}
        if (images is None) or (texts is None):
            raise RuntimeError('Missing Image OR Text in the input')

        """
            total_prediction (dict):
                # image
                unmasked_image_feats,
                unmasked_image_feat,
                if mask_image:
                    masked_image_feats,

                # text
                unmasked_text_feat
                unmasked_text_feats
                if text_mask:
                    masked_text_feat
                    masked_text_feats
        """
        # return dict
        total_prediction = dict()

        # 全局特征
        image_ret = self.encode_image(images, self.image_mask)

        # image_feats: NLD, image_feat: ND
        unmasked_image_feats = image_ret['unmasked_image_feats']
        unmasked_image_feat = unmasked_image_feats[:, 0]
        total_prediction['unmasked_image_feats'] = unmasked_image_feats
        total_prediction['unmasked_image_feat'] = unmasked_image_feat

        text_output = self.encode_text(texts)
        unmasked_text_feat = text_output["text_feat"]
        unmasked_text_feats = text_output['text_feats']
        total_prediction['unmasked_text_feat'] = unmasked_text_feat
        total_prediction['unmasked_text_feats'] = unmasked_text_feats

        if self.image_mask:
            # image_feats: NLD
            masked_image_feats = image_ret['masked_image_feats']
            total_prediction['masked_image_feats'] = masked_image_feats

        if self.text_mask:
            #with torch.no_grad():
            masked_texts = {'input_ids': batch['mlm_ids'],
                    'attention_mask': batch['attention_mask']}
            masked_text_output = self.encode_text(masked_texts)
            masked_text_feats = masked_text_output['text_feats']
            total_prediction['masked_text_feats'] = masked_text_feats

        return total_prediction
