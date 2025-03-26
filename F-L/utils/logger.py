import logging
import os
import sys
import os.path as op


def setup_logger(name, save_dir, if_train, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    #否则，它配置了一个 StreamHandler 以将消息记录到控制台。将日志级别设置为 DEBUG，并应用了特定的日志消息格式。
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
   #如果指定的 save_dir 不存在，则创建该目录。
    if not op.exists(save_dir):
        print(f"{save_dir} is not exists, create given directory")
        os.makedirs(save_dir)
   #对于训练（if_train=True），它创建一个新的文件处理程序（FileHandler）以写入训练日志。如果目录存在，将覆盖现有的 "train_log.txt" 文件。
    if if_train:
        fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
   #对于测试（if_train=False），它创建一个文件处理程序以将测试日志追加到现有的 "test_log.txt" 文件中。
    else:
        fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger