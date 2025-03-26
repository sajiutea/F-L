import logging
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
#from datasets.sampler_ddp import RandomIdentitySampler_DDP
from .dataset import ImageTextMLMDataset

#from utils.comm import get_world_size




#build_transforms 函数用于构建数据预处理的转换，根据是否是训练集以及是否进行数据增强选择不同的预处理策略。
def build_transforms(img_size=(224, 224)):
    l_transforms = transforms.Compose([
        transforms.RandomCrop(224),        
        transforms.ToTensor(),           
    ])
    f_transforms = transforms.Compose([          
        transforms.RandomCrop(224),       
        transforms.ToTensor(),           
    ])
    '''
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
       # transforms.Normalize(mean=[0.4978], std=[0.2449])])
    '''
    return f_transforms,l_transforms

#collate 函数用于将一个批次（batch）的样本数据按照其包含的键（keys）进行整理，转换成模型可以处理的张量字典。
def collate(batch):
    #获取所有样本数据中包含的键，去重得到键的集合。
    keys = set([key for b in batch for key in b.keys()])
    # 将每个样本数据的键值对按键进行整理，得到一个字典，字典的键是数据中所有可能的键，对应的值是一个列表，包含了每个样本数据中该键对应的值。
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    #用于存储整理后的批次数据的张量形式。
    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):#如果值的类型是整数（int），将其转换为 PyTorch 张量并加入 batch_tensor_dict。
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):#如果值的类型是 PyTorch 张量，将其叠加（stack）成一个新的张量并加入 batch_tensor_dict。
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


# build_dataloader 函数用于构建训练集和验证集（或测试集）的 DataLoader。
def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("MMR.dataset")

    if args.training:
        # 构建训练集的图像和文本的转换（transform）对象。
        f_transforms,l_transforms = build_transforms(img_size=args.img_size)


        train_set = ImageTextMLMDataset(
            f_transforms,
            l_transforms,
            text_length=args.text_length)

        # 根据采样器类型构建训练数据加载器 (train_loader)。

        if args.distributed == True:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=False,
                                      sampler=train_sampler,
                                      collate_fn=collate)
            return train_loader, train_sampler
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      collate_fn=collate)
            return train_loader
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))


