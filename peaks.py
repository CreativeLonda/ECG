from filter import filtered_ecg
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

fs = 500
diff_ecg = np.diff(filtered_ecg)
squared_ecg = diff_ecg ** 2
window_size = int(0.1 * fs)
ma_ecg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')

height_threshold = 0.5 * np.max(ma_ecg)

min_distance = int(0.25 * fs)

peaks, _ = find_peaks(ma_ecg, height=height_threshold, distance=min_distance)

r_peaks = []

search_window = int(0.08 * fs)

for p in peaks:
    start = max(p - search_window, 0)
    end = min(p + search_window, len(filtered_ecg))
    real_r = start + np.argmax(filtered_ecg[start:end])
    r_peaks.append(real_r)

r_peaks = np.array(r_peaks)

plt.figure(figsize=(15,4))
plt.plot(filtered_ecg, label='Filtered ECG')
plt.plot(r_peaks, filtered_ecg[r_peaks], 'ro', label='R-peaks')
plt.legend()
plt.show()

rr_intervals = np.diff(r_peaks) / fs  
fs = 500  

p_peaks = []
q_peaks = []
s_peaks = []
t_peaks = []

p_window = int(0.2 * fs)   
q_window = int(0.05 * fs)  
s_window = int(0.05 * fs)
t_window = int(0.4 * fs)   

for r in r_peaks:
    # P-wave: search backward from R peak
    start = max(r - p_window, 0)
    end = r
    if start < end:
        p_idx = start + np.argmax(filtered_ecg[start:end])
        p_peaks.append(p_idx)

    # Q-wave: small minimum before R
    start = max(r - q_window, 0)
    end = r
    if start < end:
        q_idx = start + np.argmin(filtered_ecg[start:end])
        q_peaks.append(q_idx)

    # S-wave: small minimum after R
    start = r
    end = min(r + s_window, len(filtered_ecg))
    if start < end:
        s_idx = start + np.argmin(filtered_ecg[start:end])
        s_peaks.append(s_idx)

    # T-wave: peak after R
    start = r + s_window
    end = min(r + t_window, len(filtered_ecg))
    if start < end:
        t_idx = start + np.argmax(filtered_ecg[start:end])
        t_peaks.append(t_idx)

# Convert to arrays
p_peaks = np.array(p_peaks)
q_peaks = np.array(q_peaks)
s_peaks = np.array(s_peaks)
t_peaks = np.array(t_peaks)

# Plot ECG with detected peaks
plt.figure(figsize=(15, 5))
plt.plot(filtered_ecg, label='Filtered ECG', color='black')
plt.plot(r_peaks, filtered_ecg[r_peaks], 'ro', label='R peaks')
plt.plot(p_peaks, filtered_ecg[p_peaks], 'go', label='P peaks')
plt.plot(q_peaks, filtered_ecg[q_peaks], 'mo', label='Q peaks')
plt.plot(s_peaks, filtered_ecg[s_peaks], 'co', label='S peaks')
plt.plot(t_peaks, filtered_ecg[t_peaks], 'yo', label='T peaks')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('ECG Waveform with Detected P, Q, R, S, T Peaks')
plt.show()
