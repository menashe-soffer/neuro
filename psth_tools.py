import numpy as np
import matplotlib.pyplot as plt

from rdm_tools import read_data_single_two_sessions_single_epoch


def calc_psth_basic(contact_list, subject_ids, ACTIVE_CONTACTS_ONLY):

    SPS = 4
    epoch_count = len(contact_list[0]['first'])  # len(epoch_subsets)
    _, active_contact_mask, _ = read_data_single_two_sessions_single_epoch(contact_list, subject_ids, v_samp_per_sec=SPS, active_contacts_only=ACTIVE_CONTACTS_ONLY, esel=0, cmprs=False)
    for i_esel in range(1, epoch_count):
        print(i_esel)
        data, active_contact_mask_, _ = read_data_single_two_sessions_single_epoch(contact_list, subject_ids,
                                                                                v_samp_per_sec=SPS,
                                                                                active_contacts_only=ACTIVE_CONTACTS_ONLY,
                                                                                esel=i_esel, cmprs=False)
        active_contact_mask *= active_contact_mask_
        #
        maxes = data[:, :, :int(3*SPS)].max(axis=0).max(axis=-1)
        thd = np.quantile(maxes, 0.8)
        print('>>>>', data.shape, active_contact_mask_.shape, thd)
        active_contact_mask *= (maxes > thd)
        #
        print(i_esel, active_contact_mask_.sum(), active_contact_mask.sum())
    #
    fig, ax = plt.subplots(1+3, 2, sharex=True)
    [ax[0, i].grid(True) for i in range(2)]
    [ax[0, i].set_ylim((0.9, 1.1)) for i in range(2)]
    total_PSTH = None
    for i_esel in range(epoch_count):
        print(i_esel)
        data_mat, _, _ = read_data_single_two_sessions_single_epoch(contact_list, subject_ids, v_samp_per_sec=SPS,
                                                                    active_contacts_only=ACTIVE_CONTACTS_ONLY,
                                                                    esel=i_esel, cmprs=False)
        tscale = (np.arange(data_mat.shape[-1]) - SPS) / SPS
        data_mat = data_mat[:, active_contact_mask]
        for i in range(2):
            print(data_mat.shape)
            epoch_str = 'epochs {}-{} (avg)'.format(i*9 + i_esel*3, i*9 + i_esel*3 + 2)
            ax[0, i].plot(tscale, data_mat[i].mean(axis=0), label=str(epoch_str))
            ax[i_esel+1, i].plot(tscale, data_mat[i].T, linewidth=0.5)
            ax[i_esel + 1, i].set_ylim((0, 2.5))
            ax[i_esel + 1, i].grid(True)
            ax[i_esel + 1, i].set_ylabel(epoch_str + '\nall contacts')
            if total_PSTH is None:
                total_PSTH = data_mat[i].mean(axis=0)
                avg_cnt = 1
            else:
                total_PSTH += data_mat[i].mean(axis=0)
                avg_cnt += 1

    [ax[0, i].legend() for i in range(2)]
    [ax[0, i].plot(tscale, total_PSTH / avg_cnt, linestyle='dashed', c='k', linewidth=3) for i in range(2)]


    plt.show()

