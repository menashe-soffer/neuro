import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def pierson(x, y, remove_nans=False):

    if remove_nans:
        rmv_cols = np.concatenate((np.argwhere(np.isnan(x)).flatten(), np.argwhere(np.isnan(y)).flatten()))
        if rmv_cols.size > 0:
            x, y = np.copy(x), np.copy(y)
            rmv_cols = np.sort(np.unique(rmv_cols))
            while rmv_cols.size > 0:
                x = np.concatenate((x[:rmv_cols[0]], x[rmv_cols[0]+1:]))
                y = np.concatenate((y[:rmv_cols[0]], y[rmv_cols[0]+1:]))
                rmv_cols = rmv_cols[1:] - 1

    x1, y1 = x - x.mean(), y - y.mean()
    return (x1 * y1).sum() / (np.linalg.norm(x1) * np.linalg.norm(y1) + 1e-16)

def calc_rdm(x, y):

    assert x.shape == y.shape
    assert x.ndim == 2
    num_digits = x.shape[0]
    rdm = np.zeros((num_digits, num_digits))
    for i1 in range(num_digits):
        for i2 in range(num_digits):
            rdm[i1, i2] = pierson(x[i1], y[i2])

    return rdm


def relative_codes(cmat, first=0, last=-1, remove_diag=True, normalize=False):

    # converting the correlation matrix ("rdm") into relational coding
    # remove diagonal: if True, the diagonal is first removed
    # remove columns: if default values are overridden, it will be used to select which columns in the original matrix participating in the codes

    assert cmat.shape[0] == cmat.shape[1]
    num_cw = cmat.shape[0]

    code = np.copy(cmat)
    if normalize:
        for i_cw in range(num_cw):
            code[i_cw] = code[i_cw] / code[i_cw, i_cw]

    if remove_diag:
        for i in range(num_cw):
            code[i, i] = np.nan

    return code[:, first:last]



# statistical model for relational codes

sess1_resp_amp, sess_1_resp_noise = 1, 0.4
sess2_resp_amp, sess_2_resp_noise = 1, 0.4
resp_1to2_dev = 0.4
resp_count = 100
non_rest_count = 100
non_resp_noise = 5.5
num_digits = 12


# basic response vectors
resp1 = np.random.uniform(0, 1, resp_count)
resp1 = sess1_resp_amp * resp1 / np.linalg.norm(resp1)
resp_dev = np.random.uniform(0, 1, resp_count)
resp_dev = sess1_resp_amp * resp_dev / np.linalg.norm(resp_dev)
resp2 = (1 - resp_1to2_dev) * resp1 + resp_1to2_dev * resp_dev
resp2 = sess2_resp_amp * resp2 / np.linalg.norm(resp2)


# now expand over time
resp1 = np.tile(resp1, (num_digits, 1))
#resp1 = tmp + np.random.uniform(0, 0.5 * np.sqrt(1 / resp1.size), tmp.shape)
resp2 = np.tile(resp2, (num_digits, 1))
#resp2 = tmp + np.random.uniform(0, 0.5 * np.sqrt(1 / resp1.size), tmp.shape)
for i in range(1, num_digits):
    resp1[i] = resp1[i-1] + np.random.uniform(0, 0.5 * np.sqrt(1 / resp1.shape[-1]), resp1.shape[-1])
    resp1[i] = sess1_resp_amp * resp1[i] / np.linalg.norm(resp1[i])
    resp2[i] = resp2[i-1] + np.random.uniform(0, 0.5 * np.sqrt(1 / resp2.shape[-1]), resp2.shape[-1])
    resp2[i] = sess2_resp_amp * resp2[i] / np.linalg.norm(resp2[i])

# create two "epochs"
resp1a = resp1 + np.random.normal(0, sess_1_resp_noise * np.sqrt(1 / resp1.size), resp1.shape)
resp1b = resp1 + np.random.normal(0, sess_1_resp_noise * np.sqrt(1 / resp1.size), resp1.shape)
resp2a = resp2 + np.random.normal(0, sess_2_resp_noise * np.sqrt(1 / resp2.size), resp2.shape)
resp2b = resp2 + np.random.normal(0, sess_2_resp_noise * np.sqrt(1 / resp2.size), resp2.shape)



noise_contacts_1a = (non_resp_noise / 100) * np.random.uniform(0, 1, (num_digits, 1000))
noise_contacts_1b = (non_resp_noise / 100) * np.random.uniform(0, 1, (num_digits, 1000))
noise_contacts_2a = (non_resp_noise / 100) * np.random.uniform(0, 1, (num_digits, 1000))
noise_contacts_2b = (non_resp_noise / 100) * np.random.uniform(0, 1, (num_digits, 1000))


for non_rest_count in np.arange(start=0, stop=1000, step=400):
    resp1a_full = np.concatenate((resp1a, noise_contacts_1a[:, :non_rest_count]), axis=1)
    resp1b_full = np.concatenate((resp1b, noise_contacts_1b[:, :non_rest_count]), axis=1)
    resp2a_full = np.concatenate((resp2a, noise_contacts_2a[:, :non_rest_count]), axis=1)
    resp2b_full = np.concatenate((resp2b, noise_contacts_2b[:, :non_rest_count]), axis=1)

    rdm1 = calc_rdm(resp1a_full, resp1b_full)
    rdm2 = calc_rdm(resp2a_full, resp2b_full)
    rdm_x = (calc_rdm(resp1a_full, resp2a_full) + calc_rdm(resp1b_full, resp2b_full)) / 2
    code1, code2 = relative_codes(rdm1), relative_codes(rdm2)

    R1 = relative_codes(calc_rdm(resp1a_full, resp1b_full))
    R2 = relative_codes(calc_rdm(resp2a_full, resp2b_full))

    rep_pcors = np.zeros((num_digits, num_digits))
    for digit_1 in range(num_digits):
        for digit_2 in range(num_digits):
            v1, v2 = R1[digit_1], R2[digit_2]
            rep_pcors[digit_1, digit_2] = pierson (v1, v2, remove_nans=True) #(v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)

    # print(rdm_x)
    # print(rep_pcors)
    fig, ax = plt.subplots(4, 1, figsize=(16, 10))
    ax[0].plot(resp1a_full.T)
    ax[1].plot(resp1b_full.T)
    ax[2].plot(resp2a_full.T)
    ax[3].plot(resp2b_full.T)


    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('{} contacts'.format(resp1a_full.shape[-1]))
    sns.heatmap(np.round(rdm1, decimals=2), ax=ax[0, 0], vmin=-1, vmax=1, annot=True)
    sns.heatmap(np.round(rdm2, decimals=2), ax=ax[1, 0], vmin=-1, vmax=1, annot=True)
    sns.heatmap(np.round(rdm_x, decimals=2), ax=ax[0, 1], vmin=-1, vmax=1, annot=True)
    sns.heatmap(np.round(rep_pcors, decimals=2), ax=ax[1, 1], vmin=-1, vmax=1, annot=True)

plt.show()

print('\n\n')

a, b = np.ones(10), np.ones(10)
a[5] += 0.1
b[5] -= 0.1
print('a= ', a)
print('b= ', b)
print(pierson(a, b))

