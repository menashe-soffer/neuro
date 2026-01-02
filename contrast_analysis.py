import numpy as np
import matplotlib.pyplot as plt

def eval_contrast(data_1, data_2, USE='ALL'):
    
    data = np.concatenate((data_1, data_2), axis=0)
    num_epochs, num_contacts, rdm_size = data.shape
    scores = np.zeros(num_contacts)
    for i_cntct in range(num_contacts):
        x = data[:, i_cntct]
        x = x.mean(axis=0)[np.newaxis, :] # use avg of all epochs (is it a good thing?????)
        x /= np.linalg.norm(x)
        USE_PIERSON = False
        if USE_PIERSON:
            r = np.zeros((rdm_size, rdm_size))
            for i_epoch in range(x.shape[0]):
                xx = x[np.newaxis, i_epoch] - x[np.newaxis, i_epoch].mean()
                xx = xx / xx.std()
                r += xx.T @ xx
        else:
            r = x.T @ x
        r = r[1:-1, 1:-1]
        r = r[1:, 1:]
        N = r.shape[0]
        f_wcc = np.diag(r).sum() / N
        f_occ = (r.sum() - np.diag(r).sum()) / (N * (N - 1))
        scores[i_cntct] = f_wcc - f_occ
        #scores[i_cntct] = f_wcc / (f_occ + 1e-12)
    
    if USE == 'ALL':
        mask = np.ones(num_contacts, dtype=bool)
    if USE == 'HIGH_RESP':
        thd = np.quantile(scores, 0.95)
        mask = scores > thd
    if USE == 'RESP':
        thd = np.quantile(scores, 2/3)
        mask = scores > thd
    if USE == 'NON_RESP':
        thd = np.quantile(scores, 2/3)
        mask = scores < thd
    
    return mask
        