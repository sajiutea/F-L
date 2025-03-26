import logging
import time
import torch
from utils.meter import AverageMeter
from utils.comm import get_rank
import os


def do_train(start_epoch, args, model, train_loader, train_sampler, optimizer, scheduler, checkpointer,device): 
    log_period = args.log_period
    eval_period = args.eval_period
    logit_scale = torch.ones([]) * (1 / args.temperature)
    num_epoch = args.num_epoch
    arguments = dict()
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0


    logger = logging.getLogger("MMR.train")
    logger.info('start training')
    # meters 是一个用于记录各种指标的字典(平均值更新）
    meters = {
        "loss": AverageMeter(),
        "cpa_loss": AverageMeter(),
        "cta_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "mlm_acc": AverageMeter(),          
        #"mim_loss": AverageMeter(),  
        #"mlm_error_loss": AverageMeter(),          
    }

    # train这是主要的训练循环，遍历每个 epoch。
    for epoch in range(start_epoch, num_epoch + 1):
        '''
        if epoch>6:
            os.system('shutdown -s -t 360')    
        '''
        start_time = time.time()
        train_sampler.set_epoch(epoch)
        for meter in meters.values():
            meter.reset()
        print(f'-------------epoch{epoch}--------------', epoch)
        model.train()
        # 遍历每个 mini-batch
        for n_iter, batch in enumerate(train_loader):
            # 清空梯度
            optimizer.zero_grad()

            # 将 mini-batch 中的数据移到指定的设备上（通常是 GPU）
            batch = {k: v.to(device) for k, v in batch.items()}

            # 调用模型进行前向计算，计算总的损失。
            itc_loss,mlm_loss,mlm_acc,cta_loss,cpa_loss = model(batch)
            total_loss = itc_loss+mlm_loss+cta_loss+cpa_loss # 总损失
            torch.cuda.synchronize()

            # 更新各个监测指标的值。
            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            # get 方法是用于获取字典中指定键的值,若是不存在此键，就返回默认值0
            meters['itc_loss'].update(itc_loss.item(), batch_size)
            meters['mlm_loss'].update(mlm_loss.item(), batch_size)
            meters['cpa_loss'].update(cpa_loss.item(), batch_size)
            meters['cta_loss'].update(cta_loss.item(), batch_size)
            meters['mlm_acc'].update(mlm_acc.item(), 1)            
           # meters['mim_loss'].update(mim_loss,batch_size)
            #meters['mlm_error_loss'].update(mlm_error_loss.item(), batch_size)
           
            # 反向传播，更新模型参数。
            total_loss.backward()
            optimizer.step()

            # 每隔一定的迭代周期输出训练信息，包括当前 epoch、迭代次数、损失和准确率等信息。
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if k!='num_lateral':
                        if v.avg>0:
                            info_str += f", {k}: {v.avg:.4f}"  
                    else:
                        info_str += f", {k}: {v.sum:.4f}"  
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        # 调整学习率
        scheduler.step()
        # 码检查当前进程的 rank 是否为 0（主进程），如果是，就输出当前 epoch 完成的信息，包括每个 mini-batch 的平均时间和训练速度。
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))

        # 每隔一定的 epoch 周期进行模型的验证（测试）并保存最佳模型。
        '''
        if epoch<4 and (epoch - 1) % eval_period == 0:
            if get_rank() == 0:  # 是否是主进程
                logger.info("Save model - Epoch: {}".format(epoch))
                arguments["epoch"] = epoch
                checkpointer.save("epoch{}".format(epoch), **arguments)
        '''
        if epoch>=1:
            if get_rank() == 0:  # 是否是主进程
                logger.info("Save model - Epoch: {}".format(epoch))
                arguments["epoch"] = epoch
                checkpointer.save("epoch{}".format(epoch), **arguments)       
        
        
        torch.cuda.empty_cache()