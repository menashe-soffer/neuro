import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import tqdm
import pickle
import sys

from rdm_tools_new import *
from paths_and_constants import *
from data_availability_new import data_availability, contact_list_services


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


class ContrastiveProjector(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim=100, output_dim=100):
        super(ContrastiveProjector, self).__init__()
        # Layer 1: From MNE input to 100
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        # Layer 2: From 100 to 100 (The embedding/projection head)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flattening ensures we handle [batch, channels, time] correctly
        x = x.view(x.size(0), -1) 
        
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Standard for SimCLR/Triplet: Project to a unit hypersphere
        return torch.nn.functional.normalize(x, p=2, dim=1)


class ieegTripletDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_tensor, mode='triplet'):
        """
        data_tensor: Shape (I1_max, Time_points, I2_max)
        We assume i1 is in range [2, 12] as per your requirement.
        """
        self.data = data_tensor
        # Based on your logic: i1 is index 2 to 12
        self.i_digit_indices = list(range(2, 13)) 
        self.i_epoch_range = data_tensor.shape[0]
        self.mode = mode

    def __len__(self):
        # We can define an epoch as one pass through all i2 categories
        return self.i_epoch_range * len(self.i_digit_indices)

    def __getitem__(self, i_anchor):
        
        assert self.mode == 'triplet'
        
        # 1. Pick Anchor
        i_epoch = int(i_anchor / len(self.i_digit_indices))# i_anchor
        i_digit = self.i_digit_indices[i_anchor % len(self.i_digit_indices)]# random.choice(self.i_digit_indices)
        #print(i_anchor, i_epoch, i_digit)
        anchor = self.data[i_epoch, :, i_digit]

        # 2. Pick Positive (Same i_digit, different i_epoch1)
        i_epoch_pos = random.choice([idx for idx in np.arange(self.i_epoch_range) if idx != i_epoch])
        positive = self.data[i_epoch_pos, :, i_digit]
        assert (i_epoch_pos != i_epoch)

        # 3. Pick Negative (Different i2, any i1)
        i_digit_neg = random.choice([idx for idx in self.i_digit_indices if idx != i_digit])
        i_epoch_neg = random.choice(np.arange(self.i_epoch_range))
        negative = self.data[i_epoch_neg, :, i_digit_neg]
        assert (i_digit_neg != i_digit)

        return anchor, positive, negative


def train_a_net():    

    V_SAMP_PER_SEC = 10
    V_SAMP_PER_SEC_RDM = 1
    # AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets
    MIN_TGAP, MAX_TGAP = 60, 144#144, 336#24, 48
    CONTACT_SPLIT = None # None: use all, 0: even contacts only, 1: odd contacts only
    #event_type = 'CNTDWN' # one of: 'CNTDWN', 'RECALL', 'DSTRCT', 'REST'
    #
    RAW_EPOCH_AVG = 1
    WITHIN_SESSION_PROCESS = True
    if WITHIN_SESSION_PROCESS:
        MIN_TGAP, MAX_TGAP = 1, 1000
        # WITHIN_SESSION_SEGMENT_SIZE, WITHIN_SESSION_PAIR_IDX = 6, 1
        #WITHIN_SESSION_SEGMENT_SIZE, WITHIN_SESSION_PAIR_IDX = 3, 1
        EPOCHS_TO_READ = 18
    else:
        EPOCHS_TO_READ = 18
        # WITHIN_SESSION_SEGMENT_SIZE = 6#len(AVG_MANY_EPOCHS)


    data_availability_obj = data_availability()
    epoch_subsets = [[i*RAW_EPOCH_AVG, (i+1)*RAW_EPOCH_AVG-1] for i in range(int(EPOCHS_TO_READ / RAW_EPOCH_AVG))]
    epoch_subsets = ['e{}-e{}'.format(i1, i2) for (i1, i2) in epoch_subsets]
    
    
    # prepare contact list
    # stage 1: find suitable contacts
    
    list_1C, list_2C = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                           proc_type='gamma_c_60_160', event_list=['CNTDWN'], 
                                                                                           num_epochs=18, enforce_first=True, single_session=WITHIN_SESSION_PROCESS)

    contact_info = data_availability_obj.get_contact_info(list_1C)
    boundary_sec = np.arange(start=-2, stop=12+1e-6, step=1/V_SAMP_PER_SEC_RDM)
    input_tscale = boundary_sec[:-1]

    # read data
    data_1C, cntct_mask = read_epoch_files_by_list(list_1C, first_epoch=0, last_epoch=EPOCHS_TO_READ, boundary_sec=boundary_sec, random_shift=False)
    data_1C = data_1C[:, cntct_mask, :].astype(np.float32)
    contact_info = [contact_info[i] for i in np.argwhere(cntct_mask).flatten().astype(int)]
    
    num_epochs, num_contacts, num_digits = data_1C.shape
    
    
    print(data_1C.shape, input_tscale.shape)
    # normalize
    # for i_cntct in range(data_1C.shape[1]):
    #     data_1C[:, i_cntct, :] -= data_1C[:, i_cntct, :].mean()
    data_1C = np.log10(data_1C + 1e-10)
    for i_epoch in range(num_epochs):
        for i_digit in range(num_digits):
            data_1C[i_epoch, :, i_digit] = (data_1C[i_epoch, :, i_digit] - data_1C[i_epoch, :, i_digit].mean()) / (data_1C[i_epoch, :, i_digit].std() + 1e-8)
    
    
    # 1. Setup Data
    dataset = ieegTripletDataset(data_1C)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Setup Model & Loss
    model = ContrastiveProjector(input_dim=num_contacts) # The class we built earlier
    criterion = torch.nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 3. Training Loop snippet
    num_epochs = 500
    loss_trace = np.zeros(num_epochs)
    use_tqdm = sys.stdout.isatty()
    for i_epoch in range(num_epochs):
        epoch_loss = 0
        steps = 0
        for anchors, positives, negatives in tqdm.tqdm(loader, disable=not use_tqdm):
            # Forward pass
            a_embed = model(anchors)
            p_embed = model(positives)
            n_embed = model(negatives)
            
            # Calculate loss
            loss = criterion(a_embed, p_embed, n_embed)
            #loss_trace[i_epoch] = loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print(i_epoch, loss)
            epoch_loss += float(loss)
            steps += 1
        
        epoch_loss /= steps
        print('epoch {}  loss {:6.4f}'.format(i_epoch, epoch_loss))
        loss_trace[i_epoch] = epoch_loss
        
        wgt_fname = os.path.join(IDXS_FOLDER, 'weights.pth')
        torch.save(model.state_dict(), wgt_fname)
        
        weights = model.fc1.weight.data.numpy()
        abs_weights = np.abs(weights)
        importance = np.sum(abs_weights, axis=0)
        fname = os.path.join(IDXS_FOLDER, 'contact importance')
        with open(fname, 'wb') as fd:
            pickle.dump(dict({'contact_info': contact_info, 'importance': importance}), fd)
        
        
        if i_epoch == 100:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        if i_epoch > 100:
            scheduler.step()
            
        if i_epoch > 51:
            fig, ax = plt.subplots(1)
            ax.plot(loss_trace)
            ax.plot(np.arange(25, i_epoch-24), np.convolve(loss_trace, np.ones(25) / 25, mode='same')[25:i_epoch-24], linewidth=3)
            ax.grid(True)
            mysavefig(name='train_loss', fig=fig)
    





def get_sublist_by_importance(fname=None, q=0.8):
    
    if fname is None:
        fname = os.path.join(IDXS_FOLDER, 'contact importance')
    with open(fname, 'rb') as fd:
        d = pickle.load(fd)
        contact_info = d['contact_info']
        importance = d['importance']
    mask = importance > np.quantile(importance, q=q)
    
    return mask, [contact_info[i] for i in np.argwhere(mask).flatten()]


# def select_channels_using_importance_file(contact_info, fname=None, q=0.8):
    
#     if fname is None:
#         fname = os.path.join(IDXS_FOLDER, 'contact importance')
#     with open(fname, 'rb') as fd:
#         d = pickle.load(fd)
#         importance_contact_info = d['contact_info']
#         importance = d['importance']
#     thd = np.quantile(importance, q=q)
    
#     contact_list_services_obj = contact_list_services()
#     print('selecting contacts using data in', fname)
#     mask = np.zeros(len(contact_info), dtype=bool)
#     # for i_cntct, cntct in tqdm.tqdm(enumerate(contact_info)):
#     for i_cntct in tqdm.tqdm(range(len(contact_info))):
#         cntct = contact_info[i_cntct]
#         # locate the contact in the importance file
#         idx_in_list, value = -1, -1
#         for idx, cntct_in_list in enumerate(importance_contact_info):
#             if contact_list_services_obj.is_same_contact(cntct, cntct_in_list):
#                 idx_in_list = idx
#                 value = importance[idx_in_list]
#         mask[i_cntct] = value > thd
    
#     #print(mask.sum())
#     return mask

    

def select_channels_by_regions(contact_info, region_list=[], soft=False):
        
    mask = np.zeros(len(contact_info), dtype=bool)
    for i_cntct, cntct in enumerate(contact_info):
        ok1 = cntct['location'][0]['region'] in region_list
        ok2 = cntct['location'][1]['region'] in region_list
        mask[i_cntct] = (ok1 or ok2) if soft else (ok1 and ok2)
    
    return mask, [contact_info[i] for i in np.argwhere(mask).flatten()]





if __name__ == '__main__':
    
    #train_a_net()
    fname = os.path.join(IDXS_FOLDER, 'contact importance')
    
    with open(fname, 'rb') as fd:
        d = pickle.load(fd)
        contact_info = d['contact_info']
    mask_1, contact_info_1 = get_sublist_by_importance(fname=fname)
    mask_2, contact_info_2 = select_channels_by_regions(contact_info=contact_info, region_list=['fusiform-L', 'fusiform-L'])
    print(mask_1.sum(), len(contact_info_1), mask_2.sum(), len(contact_info_2))
    
    
    data_availability_obj = data_availability()
    list_1C, list_2C = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=1, max_timegap_hrs=1000,
                                                                                       proc_type='gamma_c_60_160', event_list=['CNTDWN'], 
                                                                                       num_epochs=18, enforce_first=True, single_session=True)
    list_new, info_new = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_1C, contact_info=contact_info_1)
    print('here')
