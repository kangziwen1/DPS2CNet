from torch.utils.data import DataLoader, DistributedSampler

from networks.data.SemanticKITTI import SemanticKITTI_dataloader, collate_fn_BEV, collate_fn_BEV_test


def get_dataset(_cfg):

    if _cfg._dict['DATASET']['TYPE'] == 'SemanticKITTI':
        ds_train = SemanticKITTI_dataloader(_cfg._dict['DATASET'], 'train')
        ds_val = SemanticKITTI_dataloader(_cfg._dict['DATASET'], 'val')
        ds_test = SemanticKITTI_dataloader(_cfg._dict['DATASET'], 'test')

    _cfg._dict['DATASET']['SPLIT'] = {'TRAIN': len(ds_train), 'VAL': len(ds_val)}

    dataset = {}

    train_batch_size = _cfg._dict['TRAIN']['BATCH_SIZE']
    val_batch_size = _cfg._dict['VAL']['BATCH_SIZE']
    num_workers = _cfg._dict['DATALOADER']['NUM_WORKERS']
    # https://zhuanlan.zhihu.com/p/450912044
    train_sampler = DistributedSampler(ds_train)
    val_sampler = DistributedSampler(ds_val)

    dataset['train'] = DataLoader(ds_train, sampler=train_sampler,batch_size=train_batch_size, num_workers=num_workers,  collate_fn=collate_fn_BEV,pin_memory=True,drop_last=True)
    # dataset['val'] = DataLoader(ds_val, sampler=val_sampler,batch_size=val_batch_size, num_workers=num_workers, collate_fn=collate_fn_BEV,pin_memory=True,drop_last=True)

    # dataset['val'] = DataLoader(ds_val, sampler=val_sampler, batch_size=val_batch_size, num_workers=num_workers,
    #                             collate_fn=collate_fn_BEV, pin_memory=True, drop_last=True)
    dataset['test'] = DataLoader(ds_test, batch_size=2, num_workers=num_workers, shuffle=False, collate_fn=collate_fn_BEV_test)

    return dataset