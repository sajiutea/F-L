import torch

from .lr_scheduler import LRSchedulerWithWarmup


#这是一个用于构建优化器的函数，根据传入的参数和模型，设置不同部分的学习率和权重衰减，并返回相应的优化器对象。
def build_optimizer(args, model):
    #params = []: 用于存储不同参数组的列表。
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:#检查参数是否需要梯度。
            continue
        #根据参数名中的关键字（如 "cross"、"bias"、"classifier"、"mlm_head"），设置不同的学习率和权重衰减。
        lr = args.lr
        weight_decay = args.weight_decay

        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0        
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if "itm" in key:
            lr = args.lr * 3  #3
        #if "granularity_decoder" in key:
            #lr = args.lr 
        if "mlm_head" in key:
            lr = args.lr * args.lr_factor
        #创建参数组
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    #根据用户指定的优化器类型创建优化器对象:
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
