from bisect import bisect_right
from math import cos, pi
from torch.optim.lr_scheduler import _LRScheduler


class LRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,#一个包含学习率调整点（epoch 索引）的列表。
        gamma=0.1,#在每个调整点，学习率将乘以的因子。
        mode="step",#学习率调度模式，可以是 "step"、"exp"、"poly"、"cosine" 或 "linear"。
        warmup_factor=1.0 / 3,#在热身阶段的初始学习率乘数。
        warmup_epochs=10,#热身阶段的 epoch 数。
        warmup_method="linear",#热身阶段使用的方法，可以是 "constant" 或 "linear"。
        total_epochs=100,
        target_lr=0,
        power=0.9,#在某些学习率模式中使用的幂参数。
        last_epoch=-1,#最后一个 epoch 的索引，默认为 -1。
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones),
            )
        if mode not in ("step", "exp", "poly", "cosine", "linear"):
            raise ValueError(
                "Only 'step', 'exp', 'poly' or 'cosine' learning rate scheduler accepted"
                "got {}".format(mode)
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.mode = mode
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.power = power
        super().__init__(optimizer, last_epoch)
   #根据选择的学习率调度模式计算当前 epoch 的学习率。
   #考虑了热身阶段、步进衰减、指数衰减、线性衰减、多项式衰减和余弦退火。
    def get_lr(self):

        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        if self.mode == "step":
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]

        epoch_ratio = (self.last_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )

        if self.mode == "exp":
            factor = epoch_ratio
            return [base_lr * self.power ** factor for base_lr in self.base_lrs]
        if self.mode == "linear":
            factor = 1 - epoch_ratio
            return [base_lr * factor for base_lr in self.base_lrs]

        if self.mode == "poly":
            factor = 1 - epoch_ratio
            return [
                self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                for base_lr in self.base_lrs
            ]
        if self.mode == "cosine":
            factor = 0.5 * (1 + cos(pi * epoch_ratio))
            return [
                self.target_lr + (base_lr - self.target_lr) * factor
                for base_lr in self.base_lrs
            ]
        raise NotImplementedError
