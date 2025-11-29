from data import ecg
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(ecg)
plt.title("Raw ECG Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()