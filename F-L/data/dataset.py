# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:59:50 2023

@author: 阿九
"""
from torch.utils.data import Dataset
#import tokenizers
import os
import pandas as pd
from PIL import Image
import torch 
import random
from transformers import AutoTokenizer
import cv2
import numpy as np
#import re
#from nltk.tokenize import RegexpTokenizer

entities = ['abnormality', 'abscess', 'aerate', 'aorta', 'atelectasis', 'bronchiectasis', 'calcification', 'cardiomediastinal', \
            'cardiomegaly', 'catheter', 'chf', 'collapse', 'congestion', 'consolidation', 'contour', 'COPD', \
            'deformity', 'dilation', 'distention', 'edema', 'effusion', 'embolism', 'emphysema', 'engorgement', \
            'fibrosis', 'fracture', 'granuloma', 'hernia', 'hilar', 'hyperinflate', 'hemidiaphragm', 'infiltrate', \
            'mass','nodule', 'obscure', 'opacity', 'perihilar', 'pneumonia', 'pneumothorax', 'sarcoidosis', \
            'silhouette', 'thickening', 'tuberculosis', 'vasculature']


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img


def get_imgs(img_path, scale, transform=None):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x,scale)
    img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(img)
    return img
def clean_text(x):
    # pick impression, findings, last_paragraph    
    
    # use space instead of newline
    captions = x.replace("\n", " ")

    # split sentences这一系列步骤将完整的医学报告文本有效地分割成单独的句子
    splitter = re.compile("[0-9]+\.")
    captions = splitter.split(captions)
    captions = [point.split(".") for point in captions]
    captions = [sent for point in captions for sent in point]

    
    study_sent = []
    # create tokens from captions
    for cap in captions:
        if len(cap) == 0:
            continue
        #将句子中的特定无效字符（\ufffd\ufffd，通常是编码错误导致的字符）替换为空格。
        cap = cap.replace("\ufffd\ufffd", " ")
        # picks out sequences of alphanumeric characters as tokens
        # and drops everything else
        #这样的处理忽略了所有的标点符号和空格
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(cap.lower())
        # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
        if len(tokens) <= 1:
            continue

        # filter tokens for current sentence
        included_tokens = []
        for t in tokens:
            t = t.encode("ascii", "ignore").decode("ascii")
            if len(t) > 0:
                included_tokens.append(t)

        if len(included_tokens) > 0:
            study_sent.append(" ".join(included_tokens))
     # separate different sentences
    series_sents = list(filter(lambda x: x != "", study_sent))
    sent = " ".join(series_sents)
       
    return sent
class ImageTextMLMDataset(Dataset):#先分词再掩码
    def __init__(self,
                 f_transform=None,
                 l_transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        
        self.f_transform = f_transform
        self.l_transform = l_transform
        self.imsize =256 #经过裁剪后，变为224
        self.text_length = text_length
        self.truncate = truncate
        self.fimages_list,self.limages_list,self.report_list = self.read_csv()
        self.bert_model_name = '/root/autodl-fs/Bioclinical_BERT'
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name,trust_remote_code=True)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.mask_generator = MaskGenerator()
            
    def read_csv(self):
        csv_path = '/root/autodl-fs/all_fl_train.csv'#'/root/mae/data/train.csv'
        df = pd.read_csv(csv_path,sep=',')
        return  df["Path_frontal"], df['Path_lateral'],df['Report'] #Path\report\GptV3.5_output  df['Path'],df['Text'] 
    
    def __len__(self):
        return len(self.fimages_list)

    # _getitem__ 方法根据指定的索引获取项目。
    def __getitem__(self, index):
        fimg_path = self.fimages_list[index]       
        fimg = get_imgs(fimg_path, self.imsize, self.f_transform)#[3,224,224]        
        
        limg_path = self.limages_list[index]      
        
        if type(limg_path) != str:
            limg = torch.zeros((3, 224, 224))
            is_lateral = False
        else:
            limg = get_imgs(limg_path, self.imsize, self.l_transform)
            is_lateral = True
              
        caption = self.report_list[index]
        '''
        gpt_output = self.gpt_list[index]
        
        if pd.isna(caption):
            caption = ' '
        if pd.isna(gpt_output):
            gpt_output = ' '
        
      
        caption += gpt_output  
        caption = caption.replace("..", ".")        
        '''
        caption_tokens = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.text_length, return_tensors='pt')
        attention_mask = caption_tokens['attention_mask']
        caption_tokens = caption_tokens['input_ids']
        
        caption_tokens = torch.flatten(caption_tokens,0)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())       
        #mim_mask = torch.tensor(self.mask_generator())
        
        ret = {                       
            'images': fimg,      
            'l_images':limg,
            'is_lateral':is_lateral,            
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'attention_mask': attention_mask,    
            #'mim_mask': mim_mask
        }
        '''
        ret = {                       
            'images': fimg,                  
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'attention_mask': attention_mask,         
            'mim_mask': mim_mask
        }
        '''
        return ret
    '''
#实现了对输入的标记序列进行随机掩码，用于语言模型任务，其概率分布遵循BERT论文中的设定。
    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.vocab['[MASK]']#mask 是特殊标记，表示要掩盖的标记。
        token_range = list(range(106, len(self.tokenizer))) # 106 ~ 28996#token_range 是一个列表，包含了所有有效标记的范围，排除了特殊标记，其范围是 1 到 len(self.tokenizer.encoder)-3。
        
        labels = []
        for i, token in enumerate(tokens):#遍历输入的标记序列 tokens。
            if 105< token < 28996:#105,28996
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.7:#以60%的概率进行掩码
                    prob /= 0.7

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask#如果掩码，则以 80% 的概率将标记更改为掩码标记。

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)#10%的概率将标记更改为随机标记。

                    # -> rest 10% randomly keep current token
                    #其余 10% 的情况下，保持标记不变。
                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:#将被掩码的标记加入 labels 列表，未被掩码的标记加入 0。
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):#如果所有的标记都未被掩码，至少保证其中一个标记被掩码，以便进行预测。
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask
        tokens = torch.tensor(tokens)
        labels = torch.tensor(labels)
        return tokens, labels
    
    '''
    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.vocab['[MASK]']#mask 是特殊标记，表示要掩盖的标记。        
        
        #entity_pos = []
        #mask_pos = []  # entity context mask position
        entity_exist = False

        for i in range(1, len(tokens)):
            if self.idxtoword[tokens[i].item()] in entities:
                entity_exist = True
                break
                
        length = len(tokens)
        labels = []
        for i, token in enumerate(tokens):  # 遍历输入的标记序列 tokens。
            if 105 < token < 28996: #105 28996
                prob = random.random()
                #实体词前面的词一定掩码，实体词以75%的概率掩码
                if self.idxtoword[token] in entities:                    
                    if prob < 0.75: #0.75
                        tokens[i] = mask
                        labels.append(token)
                    else:
                        labels.append(0)
                elif (not entity_exist and prob < 0.7):#0.7
                # 如果没有实体存在，以65%的概率掩码任意词；如果存在实体，对非实体位置以60%的概率掩码
                    tokens[i] = mask
                    labels.append(token)
                elif (entity_exist and prob < 0.65):#0.65
                    tokens[i] = mask
                    labels.append(token)
                else:
                    labels.append(0)
            else:
                labels.append(0)

        
        if all(l == 0 for l in labels):#如果所有的标记都未被掩码，至少保证其中一个标记被掩码，以便进行预测。
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask
        
        tokens = torch.tensor(tokens)
        labels = torch.tensor(labels)
        return tokens, labels
    
    '''
    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.vocab['[MASK]']#mask 是特殊标记，表示要掩盖的标记。
        
        dot = self.tokenizer.vocab['.']        
        mask_pos = []  # entity context mask position
        entity_exist = False

        for i in range(1, len(tokens)):
            if self.idxtoword[tokens[i].item()] in entities:
                entity_exist = True
                break
                        
        labels = []
        for i, token in enumerate(tokens):  # 遍历输入的标记序列 tokens。
            if 4 < token < 30522:
                prob = random.random()
                #实体词前面的词一定掩码，实体词以75%的概率掩码
                if self.idxtoword[token] in entities:       
                    
                    if prob < 0.75:
                        tokens[i] = mask
                        labels.append(token)
                    else:
                        labels.append(0)
                    
                    for j in range(1, 3):
                        if i - j <= 0:
                            break
                        else:
                            if tokens[i - j] != dot:  # 16 is "."
                                if i - j not in mask_pos:
                                    mask_pos.append(i - j)
                                if self.idxtoword[tokens[i-j]] not in entities and tokens[i - j] != mask:
                                    labels[i - j] = tokens[i-j]
                                    tokens[i - j] = mask                    
                elif (not entity_exist and prob < 0.65):
                # 如果没有实体存在，以65%的概率掩码任意词；如果存在实体，对非实体位置以60%的概率掩码
                    tokens[i] = mask
                    labels.append(token)
                elif (entity_exist and i not in mask_pos and prob < 0.6):
                    tokens[i] = mask
                    labels.append(token)
                else:
                    labels.append(0)
            else:
                labels.append(0)

        
        if all(l == 0 for l in labels):#如果所有的标记都未被掩码，至少保证其中一个标记被掩码，以便进行预测。
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask
        
        tokens = torch.tensor(tokens)
        labels = torch.tensor(labels)
        return tokens, labels
        '''

class MaskGenerator:
    def __init__(self, input_size=(224, 224), mask_patch_size=16, model_patch_size=16, mask_ratio=0.5):#0.5
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        self.rand_size_h = self.input_size[0] // self.mask_patch_size
        self.rand_size_w = self.input_size[1] // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size_h * self.rand_size_w
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size_h, self.rand_size_w))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask