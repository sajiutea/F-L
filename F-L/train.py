import os
import os.path as op
import torch
import numpy as np
import random
import time
from data.build import build_dataloader
from processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver.build import build_optimizer, build_lr_scheduler
from build import build_model
from options import get_args
from utils.comm import get_rank
from torch import distributed as dist


def set_seed(seed=0):
    ## 设置随机种子以确保实验的可重复性
    torch.manual_seed(seed)  # 设置pytorch的种子
    torch.cuda.manual_seed(seed)  # 设置cuda的种子
    torch.cuda.manual_seed_all(seed)  # 若有多块GPU，保证种子一致
    np.random.seed(seed)  # numpy库的种子
    random.seed(seed)  # python内置random块的种子
    torch.backends.cudnn.deterministic = True  # 设置 PyTorch 的 CuDNN 库为确定性模式，这将使得在相同输入情况下，CuDNN 的操作产生相同的输出。
    torch.backends.cudnn.benchmark = True  # 如果启用了这个选项，CuDNN 将会根据输入数据的大小来选择最适合的卷积算法，但这样可能会导致结果的不一致性。因此，为了保持一致性，这里将其设置为 True。


if __name__ == '__main__':
    # 获取命令行参数
    args = get_args()
    set_seed(1 + get_rank())
    #set_seed(3407)
    name = args.name
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    device=torch.device("cuda:0")
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),
                                rank=args.local_rank)  # 并行训练初始化，建议'nccl'模式
        print('world_size', dist.get_world_size())  # 打印当前进程数
        # 配置每个进程的gpu
        torch.cuda.set_device(args.local_rank)

    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()

    device_id = rank % torch.cuda.device_count()
    print(torch.cuda.device_count())
    device = torch.device(device_id)
    print(device)

    # 创建输出目录和日志记录器:
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # 使用 time.strftime 函数生成当前时间的格式化字符串，以便在输出目录中添加时间戳。
    args.output_dir = op.join(args.output_dir, f'{cur_time}_{name}')  # 构造输出路径
    print(args.output_dir)
    logger = setup_logger('MMR', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader构建数据加载器:
    train_loader, train_sampler = build_dataloader(args)
    # train_loader = build_dataloader(args)
    # 构建模型:
    model = build_model(args)
    logger.info(
        'Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))  # 用于返回张量 p 中元素的总数，即张量的元素数量

    '''
    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    '''

    model.to(device)
    # 分布式训练:
    if args.distributed == True:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True,
                                                          broadcast_buffers=False)

    # 构建优化器和学习率调度器:
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    # 检查点和评估器:
    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)

    # 恢复训练: 从第2轮恢复训练
    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch'] + 1

    # 进行训练:
    do_train(start_epoch, args, model, train_loader, train_sampler, optimizer, scheduler, checkpointer, device)
    # do_train(start_epoch, args, model, train_loader, optimizer, scheduler, checkpointer, device)
    print("finished!")
    