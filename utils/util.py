import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler

def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, omic, label, event_time, c]

def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    cluster_ids = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
    omic = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    return [img, cluster_ids, omic, label, event_time, c]

def collate_MIL_survival_sig(batch):

    img = torch.cat([item[0] for item in batch], dim=0)
    img2 = torch.cat([item[1] for item in batch], dim=0)
    img3 = torch.cat([item[2] for item in batch], dim=0)
    img4 = torch.cat([item[3] for item in batch], dim=0)
    img5 = torch.cat([item[4] for item in batch], dim=0)
    img6 = torch.cat([item[5] for item in batch], dim=0)
    img7 = torch.cat([item[6] for item in batch], dim=0)
    img8 = torch.cat([item[7] for item in batch], dim=0)
    img9 = torch.cat([item[8] for item in batch], dim=0)
    img10 = torch.cat([item[9] for item in batch], dim=0)

    coords1 = torch.cat([item[10] for item in batch], dim=0)
    coords2 = torch.cat([item[11] for item in batch], dim=0)
    coords3 = torch.cat([item[12] for item in batch], dim=0)
    coords4 = torch.cat([item[13] for item in batch], dim=0)
    coords5 = torch.cat([item[14] for item in batch], dim=0)
    index_num = [
        torch.tensor(item[15], dtype=torch.int64, device='cuda') for item in batch
    ]

    omic1 = torch.cat([item[16] for item in batch], dim=0).type(torch.FloatTensor)
    omic2 = torch.cat([item[17] for item in batch], dim=0).type(torch.FloatTensor)
    omic3 = torch.cat([item[18] for item in batch], dim=0).type(torch.FloatTensor)
    omic4 = torch.cat([item[19] for item in batch], dim=0).type(torch.FloatTensor)
    omic5 = torch.cat([item[20] for item in batch], dim=0).type(torch.FloatTensor)
    omic6 = torch.cat([item[21] for item in batch], dim=0).type(torch.FloatTensor)

    label = torch.LongTensor([item[22] for item in batch])
    event_time = np.array([item[23] for item in batch])
    c = torch.FloatTensor([item[24] for item in batch])
    return [img, img2, img3, img4, img5, img6, img7, img8, img9, img10, coords1, coords2, coords3, coords4, coords5, index_num, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c]

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, modal='coattn', batch_size=1):
    """
        return either the validation loader or training loader 
    """
    if modal == 'coattn':
        collate = collate_MIL_survival_sig
    elif modal == 'cluster':
        collate = collate_MIL_survival_cluster
    else:
        collate = collate_MIL_survival

    kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
    
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

    return loader

def set_seed(seed=7):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True