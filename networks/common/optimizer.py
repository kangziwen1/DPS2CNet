from functools import partial

import numpy as np
import torch
import torch.optim as optim
from networks.common.adan import Adan


def build_optimizer(_cfg, model):
    opt = _cfg._dict['OPTIMIZER']['TYPE']
    lr = _cfg._dict['OPTIMIZER']['BASE_LR']
    if 'MOMENTUM' in _cfg._dict['OPTIMIZER']: momentum = _cfg._dict['OPTIMIZER']['MOMENTUM']
    if 'WEIGHT_DECAY' in _cfg._dict['OPTIMIZER']: weight_decay = _cfg._dict['OPTIMIZER']['WEIGHT_DECAY']

    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               betas=(0.9, 0.999))

    elif opt == 'SGD':
        optimizer = optim.SGD(model.get_parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
    elif opt == 'Adan':
        optimizer = Adan(model.parameters(),
                         lr=lr,
                         weight_decay=weight_decay,
                         )

    return optimizer


def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size, num_gpu):
    batch_size *= num_gpu

    if num_gpu == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // num_gpu

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) / (num_epochs * iter_per_epoch)))


def build_scheduler(_cfg, optimizer):
    # Constant learning rate
    if _cfg._dict['SCHEDULER']['TYPE'] == 'constant':
        lambda1 = lambda epoch: 1
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # Learning rate scaled by 0.98^(epoch)
    if _cfg._dict['SCHEDULER']['TYPE'] == 'power_iteration':
        lambda1 = lambda epoch: (0.98) ** (epoch)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    if _cfg._dict['SCHEDULER']['TYPE'] == 'cosine':
        tMax = _cfg._dict['SCHEDULER']['T_Max']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(tMax))
    if _cfg._dict['SCHEDULER']['TYPE'] == 'cosine_warmup':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=80,
                batch_size=2,
                dataset_size=19130,
                num_gpu=2
            ),
        )

    return scheduler
