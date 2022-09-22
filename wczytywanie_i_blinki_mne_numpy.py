# !/usr/bin/env python
# coding: utf-8

import mne
from scipy.signal import find_peaks
import numpy as np
from mne.preprocessing import ICA


def wczytaj(filename):  # /mne_lib
    # po kolei dla plików z folderu

    # wczyttywanie i wstępna obróbka
    eeg = mne.io.read_raw_brainvision(filename, preload=True)

    # ## MONTAŻ

    mapa = mne.channels.make_standard_montage('standard_1020')

    eeg_mont = eeg.copy().set_montage(mapa)

    # ## REFERENCJA

    chosen_channels = ['Fp1', 'Fp2',
                       'F7', 'F3', 'Fz', 'F4', 'F8',
                       'T7', 'C3', 'Cz', 'C4', 'T8',
                       'P7', 'P3', 'Pz', 'P4', 'P8',
                       'O1', 'O2']  # (bez elektrod uszynych)

    # wybór 19 kanałóW
    eeg_20 = eeg_mont.copy()
    eeg_20.pick_channels(chosen_channels)
    # CHANNELS ORDER
    # [Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2]

    # average reference as projection
    eeg_20.set_eeg_reference('average', projection=True)

    # ## FILTROWANIE

    eeg_20.filter(1, None, l_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming',
                  n_jobs=16)
    eeg_20.notch_filter(freqs=50, filter_length='auto', phase='zero', method="iir")

    eeg_20.filter(None, 80, l_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_window='hamming',
                  n_jobs=16)

    return eeg_20


def detektor_bs(syg, option='numpy'):  # /mne_library
    if option == 'numpy':
        print("Analiza z wykorzystaniem biblioteki numpy")

        # biore Fp1 i Fp2 bo są najbliżej oczu
        syg1 = syg.copy().pick_channels(['Fp1'])
        syg1_m = syg1.get_data().flatten()  # robi macierz

        odch_stand1 = np.std(syg1_m)
        peaks1, peaks_dict1 = find_peaks(syg1_m * (-1), prominence=odch_stand1)

        syg2 = syg.copy().pick_channels(['Fp2'])
        syg2_m = syg2.get_data().flatten()  # robi macierz

        odch_stand2 = np.std(syg2_m)
        peaks2, peaks_dict2 = find_peaks(syg2_m * (-1), prominence=odch_stand2)
        blinki = []
        for i in peaks1:
            for j in peaks2:
                if i == j:
                    blinki.append(i)

        blinki = np.array(blinki)

        chosen_channels = ['Fp1', 'Fp2',
                           'F7', 'F3', 'Fz', 'F4', 'F8',
                           'T7', 'C3', 'Cz', 'C4', 'T8',
                           'P7', 'P3', 'Pz', 'P4', 'P8',
                           'O1', 'O2']

        syg_caly = syg.copy().pick_channels(chosen_channels)
        syg_caly_m = syg_caly.get_data().flatten()
        syg_caly_m = np.array(syg_caly_m)
        syg_caly_m = syg_caly_m.reshape((19, 298720))

        return syg_caly_m, blinki

    elif option == 'mne_lib':
        print("Analiza z wykorzystaniem biblioteki mne")

        # usuwanie mrugnięć z sygnału
        raw = syg

        picks_eeg = mne.pick_types(syg.info, meg=False, eeg=True, eog=False,
                                   stim=False, exclude='bads')
        method = 'fastica'  # wybrana metoda ICA

        n_components = 19
        decim = 3
        random_state = 15

        ica = ICA(n_components=n_components, method=method, random_state=random_state)

        reject = dict(mag=5e-10, grad=4000e-11)
        ica.fit(raw, picks=picks_eeg, decim=decim, reject=reject)

        # ica.plot_components(inst=syg, psd_args={'fmax': 49.})
        # ręcznie wybrany komponent z mrugnięciami do usunięcia;
        # przy większej ilości badanych można zastosowć bardziej automatyczny sposób
        ica.exclude = [0]
        reconst_syg = syg.copy()
        ica.apply(reconst_syg)

        return ica.get_sources(syg)
