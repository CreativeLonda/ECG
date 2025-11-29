from scipy.signal import butter, filtfilt
import numpy as np

fs = 500  
lowcut = 0.5
highcut = 40.0
ecg_signal = np.loadtxt(r"C:\Users\hites\OneDrive\Documents\ECG\ECG.dat")
ecg_signal = ecg_signal - np.mean(ecg_signal)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.plot(ecg_signal, label='Original')
plt.plot(filtered_ecg, label='Filtered', alpha=0.8)
plt.legend()
plt.show()