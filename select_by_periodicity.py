import numpy as np
import matplotlib.pyplot as plt
#import pywt

from rdm_tools_new import mysavefig
from paths_and_constants import *

import numpy as np
import scipy
import seaborn as sns





def contact_subset_by_periodicity(data, boundary_sec, contact_info, USE, gen_debug=False, show_only_positiove=True):

    assert data.shape[1] == len(contact_info)

    n = np.zeros((data.shape[1]), dtype=int)
    psth = np.zeros((data.shape[1], data.shape[2]))
    for i_cntct in range(data.shape[1]):
        n[i_cntct] = (data[:, i_cntct].sum(axis=-1) > 0).sum()
        psth[i_cntct] = data[:n[i_cntct], i_cntct].mean(axis=0)

    digit_start_stop = np.zeros((11, 2), dtype=int)
    digit_norm = np.zeros((data.shape[1], 10))
    for i_digit in range(11):
        tmp = np.argwhere((boundary_sec >= i_digit - 1) * (boundary_sec <= (i_digit))).flatten()
        digit_start_stop[i_digit, 0] = tmp[0]
        digit_start_stop[i_digit, 1] = tmp[-1]
    #     digit_norm[:, i_digit] = \
    #         np.linalg.norm(psth[:, digit_start_stop[i_digit, 0]:digit_start_stop[i_digit, -1] + 1], axis=1) / np.sqrt((tmp[-1] - tmp[0] + 1))

    
    # processing window for periodicity
    i0 = digit_start_stop[2 + 1, 0 + 1]
    i1 = digit_start_stop[9 + 1, -1]
    #print(data.shape, i0, i1)
    tfm = np.zeros((data.shape[1], (i1 - i0 + 1)))# * 8/8))
    xcorr = np.zeros((data.shape[1], (i1 - i0 + 1)))# * 8/8))
    # print(i0, i1, boundary_sec[i0], boundary_sec[i1], tfm.shape, xcorr.shape)
    win = np.hanning(i1 - i0 + 1)
    for i_cntct in range(data.shape[1]):
        # sig = smoothed_psth[i_cntct, i0:i1 + 1]
        sig = psth[i_cntct, i0:i1 + 1]
        sig = sig - sig.mean()
        corr = np.correlate(sig, sig, mode='full')
        corr = corr[corr.size // 2:]
        if corr[0] > 0:
            corr = corr / corr[0]
        xcorr[i_cntct] = corr
        #
        sig = np.copy(corr)#np.concat((corr[::-1][:-1], corr))
        sig = sig - sig.mean()
        sig = sig / sig.std()
        # sig = sig * np.hanning(sig.size)
        tmp = np.fft.fft(sig)
        tmp = np.real(tmp * np.conj(tmp))
        tfm[i_cntct] = np.sqrt(tmp)
    

    # profile (for start)
    i0 = digit_start_stop[0, 0]
    i1 = digit_start_stop[-1, -1] + 1
    tscale_onset = boundary_sec[i0:i1]
    i0_onset, i1_onset = i0, i1
    prof = np.zeros((data.shape[1], i1 - i0))
    for i_cntct in range(data.shape[1]):
        prof[i_cntct] = np.cumsum(psth[i_cntct, i0:i1] - np.quantile(psth[i_cntct, i0:i1], 0.05))
        prof[i_cntct] /= prof[i_cntct].max()

    # multires averaging
    levels = 5 * 2 ** np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    levels = 5 * 2 ** np.array([1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])
    lengths = ((i1 - i0) / levels).astype(int)
    rel_shfts = np.concatenate(([0], np.diff(lengths / 2).astype(int)))
    prof2d = []
    for i_cntct in range(data.shape[1]):
        results = []
        for length, rel_shft in zip(lengths, rel_shfts):
            # ker = np.ones(length) / length
            ker = np.concatenate(([1], np.zeros(length), [-1])) * 500 / length
            results.append(np.maximum(np.convolve(ker, prof[i_cntct], mode='same'), 0))    
        prof2d.append(np.array(results))



    # detect part
    fs = int(1 / np.diff(boundary_sec).mean() + 0.5)
    tmp = np.copy(xcorr)
    tmp[:, :int(fs / 10)] = 0
    # dtct = np.zeros(data.shape[1], dtype=bool)
    peak_id = np.argmax(tmp, axis=1)
    dtct = (peak_id >= fs - 1) * (peak_id <= fs + 1)
    req_bin = 350 / 50
    f = (tfm[:, int(req_bin)] + tfm[:, int(2 * req_bin)] + tfm[:, int(3 * req_bin)]) / tfm[:, :int(5 * req_bin)].sum(axis=1)
    dtct = dtct * ( f > 0.25)
    #
    dtct_start = np.zeros_like(dtct).astype(bool)
    for i_cntct in range(data.shape[1]):
        dtct_start[i_cntct] = (prof2d[i_cntct][:, :int(fs*2)].max() > 4-2) * (prof2d[i_cntct][:, int(fs*3):].max() < prof2d[i_cntct][:, :int(fs*2)].max() / 2)

    dtct_start_val = prof2d[i_cntct][:, :int(fs*3)].max() - prof2d[i_cntct][:, int(fs*3):].max()
    dtct_periodicity_val = dtct * f * 2
    #print('{}  {}   {:5.3f}    {:5.3f}  {}    {:5.3f}'.format(dtct[-1], peak_id, f[-1], dtct_periodicity_val[-1], dtct_start[-1], dtct_start_val))


    if gen_debug:
        i0 = digit_start_stop[1, 0]
        i1 = digit_start_stop[-1, -1] + 1
        #psth = psth[:, i0:i1]
        #normalized_psth = normalized_psth[:, i0:i1]
        #smoothed_psth = smoothed_psth[:, i0:i1]
        tscale = boundary_sec[i0:i1]
        for i_cntct in range(data.shape[1]):
            #dtct[i_cntct] = (prof[i_cntct] / prof[i_cntct].max())[100] >= 0.3
            if (not show_only_positiove) or dtct[i_cntct] or dtct_start[i_cntct]:
                fig, ax = plt.subplots(3, 2, figsize=(16, 8))
                ax = ax.flatten()
                ax[0].plot(tscale, psth[i_cntct][i0:i1], label='psth', linewidth=2)
                # ax[0].plot(tscale, normalized_psth[i_cntct][i0:i1], label='normed', linewidth=0.5)
                # ax[0].plot(tscale, smoothed_psth[i_cntct][i0:i1], label='smthed', linewidth=0.5)
                #ax[0].grid(True)
                ax[0].legend()
                ax[0].set_ylabel('PSTH')
                # req_bin = 350 / 50
                # f = tfm[i_cntct][int(req_bin)] / tfm[i_cntct][:int(5 * req_bin)].sum()
                ax[2].plot(tfm[i_cntct][:int(5 * req_bin)], label=str(f[i_cntct]))
                ax[2].set_ylabel('spectra')
                ax[2].legend()
                ax[1].plot(xcorr[i_cntct, :350], label=str(dtct[i_cntct]) + ' ' + str(peak_id[i_cntct]))
                ax[1].legend()
                ax[1].set_ylabel('auto-corr')
                ax[3].plot(psth[i_cntct][i0_onset:i1_onset] / psth[i_cntct][i0_onset:i1_onset].max())
                ax[3].plot(prof[i_cntct] / prof[i_cntct].max())
                idxs = np.arange(5, 500)
                #ax[3].plot(idxs, prof[i_cntct][idxs] * 0.1 * 500 / idxs)
                ax[3].set_ylabel('profile')
                ax[3].legend([str((prof[i_cntct] / prof[i_cntct].max())[100])])
                sns.heatmap(prof2d[i_cntct], ax=ax[4])
                ax[4].set_ylabel('2D profile')
                ax[5].plot(tscale_onset, prof2d[i_cntct].T, label=str(dtct_start[i_cntct]))
                ax[5].legend()
                [ax_.grid(True) for ax_ in ax]
                #
                dtct_type = 'none'
                dtct_type = 'periodic' if dtct[i_cntct] else dtct_type 
                dtct_type = 'start' if dtct_start[i_cntct] else dtct_type 
                dtct_type = 'both' if (dtct_start[i_cntct] and dtct[i_cntct]) else dtct_type 
                ext_name = '{}  {}  {}'.format(dtct_type, contact_info[i_cntct]['subject'][4:], contact_info[i_cntct]['name'])
                fig.suptitle(ext_name + '\npeak at {}  dtct {}'.format(peak_id[i_cntct], dtct[i_cntct]))
                #
                if dtct_start[i_cntct] or dtct[i_cntct] or (not show_only_positiove):
                    mysavefig(name=ext_name, subfolder='periodicity', fig=fig)
                #
                plt.close('all')

    # assert False

    if USE == 'NON_RESP':
        dtct = np.logical_not(dtct)
    contact_info = [contact_info[i] for i in np.argwhere(dtct).flatten().astype(int)]

    return contact_info, dtct, dtct_start, dtct_start_val, dtct_periodicity_val



#contact_info_, slct_mask = contact_subset_by_consistancy(data_1C_sel, boundary_sec, contact_info, USE)