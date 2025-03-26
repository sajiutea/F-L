class AverageMeter(object):
    """Computes and stores the average and current value"""
#AverageMeter 是一个用于计算平均值的辅助类。它主要用于在训练过程中追踪损失值等指标，并计算它们的平均值。
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
#update(self, val, n=1): 更新计数，传入当前值 val 和值的数量 n，更新 sum 和 count，然后重新计算平均值 avg。
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count