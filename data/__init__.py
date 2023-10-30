'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data



def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if dataset_opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
            sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            num_workers = dataset_opt['n_workers'] # * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


# def create_dataloader(train, dataset, dataset_opt, opt=None, sampler=None):
#     # gpu_ids = opt.get('gpu_ids', None)
#     gpu_ids = []
#     gpu_ids = gpu_ids if gpu_ids else []
#     num_workers = dataset_opt['n_workers'] * (len(gpu_ids)+1)
#     batch_size = dataset_opt['batch_size']
#     shuffle = True
#     if train:
#         return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
#                                            num_workers=num_workers, sampler=sampler, drop_last=True,
#                                            pin_memory=False)
#     else:
#         return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                                            num_workers=num_workers, sampler=sampler, drop_last=False,
#                                            pin_memory=False)

