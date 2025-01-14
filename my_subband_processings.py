import numpy as np
import scipy
import matplotlib.pyplot as plt # necessary for debug
import time


def split_to_subbands(data, sub_centers=[70, 90, 110, 130, 150], subs_bw=20, fs=500, show_dbg=False, dbg_markers=None):

    num_chans = data.shape[0]

    # translate to analitic signal (hilbert transform)
    print('hilbert tranform ...  ', end=" ")
    tstart = time.time()
    hx = scipy.signal.hilbert(data, axis=1)
    tend = time.time()
    print(tend - tstart)

    # split to sub-bands
    # the filter
    flen = int(4 * fs / subs_bw + 2)
    flen += flen % 2 # to have even number of taps
    b = scipy.signal.firwin(numtaps=flen, cutoff=subs_bw/2, fs=500)
    # the result array
    x = np.zeros((len(sub_centers), num_chans, data.shape[-1]), dtype=np.complex_)

    # filtering each sub-band
    for i_sub in range(len(sub_centers)):
        print('creating sub-band', i_sub, end=' ')
        tstart = time.time()
        mod = np.exp((-1j * 2 * np.pi / 500) * sub_centers[i_sub] * np.arange(hx.shape[-1]))
        t = hx * mod
        # #
        # t[:, :] = 1
        # t[:, 5000] = 1e3
        # #
        x[i_sub] = (scipy.signal.filtfilt(b, 1, np.real(t)) + 1j * scipy.signal.filtfilt(b, 1, np.imag(t)))
        tend = time.time()
        print(tend - tstart)

    # clearing the transients
    x[:, :, :2 * flen] = 0
    x[:, :, -2 * flen:] = 0

    # power
    p = np.real(x * np.conj(x))

    # visualizing
    if show_dbg:
        fig, ax = plt.subplots(6, 1, sharex=True, sharey=True)
        legstr = ['band ' + str(i+1) for i in range(len(sub_centers))]
        tscale = np.arange(p.shape[-1]) / fs
        ythd = p.max() * 0.1
        for i_chan in range(min(6, num_chans)):
            #ax[i_chan].plot(np.real(hx[i_chan]).T)
            ax[i_chan].plot(tscale, p[:, i_chan].T)
            ax[i_chan].grid(True)
            ax[i_chan].set_ylim([-0.1*ythd, ythd])
            #ax[i_chan].legend(legstr, loc='right')
            if dbg_markers is not None:
                for smp in dbg_markers:
                    ax[i_chan].plot([smp/fs, smp/fs], [-0.1*ythd, 0.5*ythd], c='k')
    plt.show()

    return x, p


def split_to_subbands1(data, sub_centers=[70, 90, 110, 130, 150], subs_bw=20, fs=500, show_dbg=False, dbg_markers=None):

    num_chans = data.shape[0]

    # filter design
    desired = (0, 0, 1, 1, 0, 0) # BPF
    tband = subs_bw / 8 # size of transition band
    filters = []
    ntaps = 77
    #fig, ax = plt.subplots(1, 5)

    for i_center, center in enumerate(sub_centers):

        bands = (0, center - (subs_bw + tband) / 2, center - (subs_bw - tband) / 2, center + (subs_bw - tband) / 2, center + (subs_bw + tband) / 2, fs / 2)
        fir_firls = scipy.signal.firls(ntaps, bands, desired, fs=fs)
        fir_remez = scipy.signal.remez(ntaps, bands, desired[::2], fs=fs)
        fir_firwin2 = scipy.signal.firwin2(ntaps, bands, desired, fs=fs)

    #     for b in [fir_firls, fir_remez, fir_firwin2]:
    #         freq, response = scipy.signal.freqz(b, 1, fs=fs)
    #         ax[i_center].plot(freq, 20 * np.log10(np.maximum(np.abs(response), 1e-5)))
    #     ax[i_center].grid(True)
    #     ax[i_center].legend(['firls', 'remez', 'firwin2', 'firwin'])
    # plt.show()

        # using the commented-out visualizing code, I have chossen firls
        filters.append(fir_firls)

    nbands = len(sub_centers)
    nchans, nsamps = data.shape
    output_data = np.zeros((nchans, nbands, nsamps))
    for i_band, filter in enumerate(filters):
        output_data[:, i_band, :] = scipy.signal.filtfilt(filter, 1, data)

    h = scipy.signal.hilbert(output_data, axis=-1)
    p = np.real(h * np.conj(h))

    return output_data, p



def plot_signals_by_chan_and_band(signal1, signal2=None, band_centers=None, bw=0, fs=500, markers=None, chan_names=None, tscope=[-1, 2]):

    nchans, nbands, nsamps = signal1.shape
    if signal2 is not None:
        assert signal1.shape == signal2.shape

    # set the time scope relative to marker
    scope_smps = np.arange(start=tscope[0] * fs, stop=tscope[1] * fs).astype(int)
    scope_time = scope_smps / fs

    #fig, ax = plt.subplots(nchans, nbands, sharex=True, sharey=True, figsize=(20, 10))
    fig, ax = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(20, 10))

    smin, smax, qmin, qmax = np.inf, -np.inf, np.inf, -np.inf

    for i_marker, marker in enumerate(markers):
        smps = marker + scope_smps
        for i_chan in range(nchans):
            ax[i_chan, 0].set_ylabel(chan_names[i_chan])
            for i_band in range(nbands):
                ax[-1, i_band].set_xlabel(str(band_centers[i_band] - bw/2) + '  -  ' + str(band_centers[i_band] + bw/2))
                ax[i_chan, i_band].plot(scope_time, signal1[i_chan, i_band, smps[0]:smps[-1]+1])
                smin = min(smin, signal1[i_chan, i_band, smps[0]:smps[-1]+1].min())
                qmin = min(qmin, np.quantile(signal1[i_chan, i_band, smps[0]:smps[-1]+1], 0.01))
                smax = max(smax, signal1[i_chan, i_band, smps[0]:smps[-1]+1].max())
                qmax = max(qmax, np.quantile(signal1[i_chan, i_band, smps[0]:smps[-1]+1], 0.99))
                if signal2 is not None:
                    ax[i_chan, i_band].plot(scope_time, signal2[i_chan, i_band, smps[0]:smps[-1]+1])
                ax[i_chan, i_band].plot([0, 0], [-1e-5, 1e-5], c='k')
                #ax[i_chan, i_band].set_xlim(tscope)
                ax[i_chan, i_band].grid(True)

        # set plot span
        qspan = qmax - qmin
        smin = max(smin, qmin - 0.2 * qspan) - 0.02 * qspan
        smax = min(smax, qmax + 0.2 * qspan) + 0.02 * qspan
        for i_chan in range(nchans):
            ax[i_chan, 0].set_ylabel(chan_names[i_chan])
            ax[i_chan, i_band].set_ylim([smin, smax])

    return fig


def calc_HFB(data, sub_centers=[70, 90, 110, 130, 150], subs_bw=20, fs=500, show_dbg=False, dbg_markers=None, chan_names=None, plot_prefix=None):

    x, p = split_to_subbands1(data, sub_centers=sub_centers, subs_bw=subs_bw, fs=fs, show_dbg=False, dbg_markers=dbg_markers)

    #
    num_markers = len(dbg_markers)
    ave_tscope = [-2, 10.5]
    ave_scope_smps = np.arange(start=ave_tscope[0] * fs, stop=ave_tscope[1] * fs).astype(int)
    # scope_time = scope_smps / fs
    num_markers_to_process = 26
    band_factors = np.array(sub_centers) / sub_centers[0]
    for i_band in range(len(band_factors)):
        x[:, i_band] *= band_factors[i_band]
        p[:, i_band] *= band_factors[i_band]

    p_ave = np.zeros((p.shape[0], p.shape[1], ave_scope_smps.size))
    for i_marker, marker in enumerate(dbg_markers[:num_markers_to_process]):
        # averaging
        smps = marker + ave_scope_smps
        p_ave += p[:, :, smps[0] : smps[-1] + 1] * (1 / len(dbg_markers))

        # plotting non-averaged signal
        #print('going to create fig for marker', i_marker)
        fig = plot_signals_by_chan_and_band(x, signal2=np.sqrt(p), band_centers=sub_centers, bw=20, fs=500, markers=[marker], chan_names=chan_names, tscope=[-1, 2])
        fig.suptitle('marker ({})  at {:7.2f}'.format(i_marker + 1, marker / fs))
        if plot_prefix is not None:
            fname = plot_prefix + 'marker_' + str(i_marker + 1)
            fig.savefig(fname)
            plt.close()
            plt.clf()
            print('saved:', fname)

    fig = plot_signals_by_chan_and_band(np.sqrt(p_ave), band_centers=sub_centers, bw=20, fs=500, markers=[np.argmin(np.abs(ave_scope_smps)).squeeze()], chan_names=chan_names, tscope=[-1, 2])
    fname = plot_prefix + 'average'
    fig.savefig(fname)
    print('saved:', fname)

    plt.show(block=False)
    plt.pause(0.1)



