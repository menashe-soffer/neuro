import numpy as np
import matplotlib.pyplot as plt

from rdm_tools_new import resample_epoch, calc_rdm, pierson, mysavefig

# rdm boundary
TSTART, TSTOP = -2, 12


def get_1s_comb(data, boundary_sec, tshift, interval=0.1):
    
    dt = np.diff(boundary_sec)[0]
    local_boundary_sec = np.arange(start=TSTART+tshift, stop=TSTOP+1e-6+tshift, step=interval)
    data = resample_epoch(data, fs=1/dt, tscale=boundary_sec[:-1], boundary_sec=local_boundary_sec)
    
    return data[:, :, ::int(1 / interval)]



def make_rdm_and_act_sets(data, boundary_sec, shift_step=0.02, shiftrange=1):
    
    epoch_rdm_set, session_rdm_set, act_set = [], [], []
    for tshift in np.arange(start=0, stop=shiftrange, step=shift_step):
        comb = get_1s_comb(data, boundary_sec=boundary_sec, tshift=tshift)
        short_rdm = np.zeros((comb.shape[-1], comb.shape[-1]))
        for i_epoch in range(comb.shape[0]):
            short_rdm += calc_rdm(data=comb[i_epoch:i_epoch+1], rdm_size=comb.shape[-1],
                                  pre_ignore=0, delta_time_smple=1)
        epoch_rdm_set.append(short_rdm)
        act_vec = comb.transpose((1, 0, 2)).reshape(comb.shape[1], comb.shape[0] * comb.shape[2])
        act_set.append(act_vec)
        session_rdm_set.append(calc_rdm(data=act_vec[np.newaxis, :, :], rdm_size=act_vec.shape[-1], 
                                        pre_ignore=0, delta_time_smple=1))
        
    return epoch_rdm_set, session_rdm_set, act_set
        



def calculate_rdm_correlations(rdm_set, full_matrix=False, offset=1):
    
    n = len(rdm_set)
    rdm_size = rdm_set[0].shape[-1]
    mask = np.eye(rdm_size, dtype=bool)
    
    if full_matrix:
        r = np.zeros((n, n))
        for i1 in range(n):
            rdm1 = rdm_set[i1][~mask].reshape(rdm_size, rdm_size-1)
            for i2 in range(i1 + 1):
                rdm2 = rdm_set[i2][~mask].reshape(rdm_size, rdm_size-1)
                r[i1, i2] = pierson(rdm1, rdm2)
                r[i2, i1] = r[i1, i2]
    else:
        r = np.zeros(n - offset)
        for i in range(n - offset):
            rdm1 = rdm_set[i][~mask].reshape(rdm_size, rdm_size-1)
            rdm2 = rdm_set[i + 1][~mask].reshape(rdm_size, rdm_size-1)
            r[i] = pierson(rdm1, rdm2)
    
    return r



def my_flow(data1, data2, boundary_sec):
    
    def single_session_flow(data, bboundary_sec):
        
        epoch_rdm_set, session_rdm_set, act_set = make_rdm_and_act_sets(data=data, boundary_sec=boundary_sec)
        epoch_rdm_corr_func = calculate_rdm_correlations(epoch_rdm_set)
        session_rdm_corr_func = calculate_rdm_correlations(session_rdm_set)
        n = len(act_set)
        offset = 1
        activation_corr_func = np.zeros(n - offset)
        for i in range(n - offset):
            activation_corr_func[i] = pierson(act_set[i], act_set[i + offset])
    
        return epoch_rdm_corr_func, session_rdm_corr_func, activation_corr_func
    
    
    fig_erdm, ax_erdm = plt.subplots(2, 1)
    fig_srdm, ax_srdm = plt.subplots(2, 1)
    fig_act, ax_act = plt.subplots(2, 1)
    
    for i_data, data in enumerate([data1, data2]):
        epoch_rdm_corr_func, session_rdm_corr_func, activation_corr_func = \
            single_session_flow(data, boundary_sec)
        for i_ax in range(2):
            ax_erdm[i_ax].plot(epoch_rdm_corr_func, label='sess ' + str(i_data))
            ax_srdm[i_ax].plot(session_rdm_corr_func, label='sess ' + str(i_data))
            ax_act[i_ax].plot(activation_corr_func, label='sess ' + str(i_data))
            
    for ax in [ax_erdm, ax_srdm, ax_act]:
        [ax[i_ax].grid(True) for i_ax in range(2)]
        ax[0].set_ylim([0.7, 1])
        ax[1].set_ylim([0, 1])
    
    fig_erdm.suptitle('epoch rangre correlation')
    fig_srdm.suptitle('session rangre correlation')
    fig_act.suptitle('session activity correlation')
    mysavefig(name='erdm', fig=fig_erdm)
    mysavefig(name='srdm', fig=fig_srdm)
    mysavefig(name='act', fig=fig_act)
  
    
    
    
    