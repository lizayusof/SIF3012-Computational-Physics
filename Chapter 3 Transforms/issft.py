#This code is taken from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html#scipy.signal.istft
#Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by 0.001 V**2/Hz of white noise sampled at 1024 Hz.

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fs = 1024
N = 10*fs
nperseg = 512
amp = 2 * np.sqrt(2)
noise_power = 0.001 * fs / 2
time = np.arange(N) / float(fs)
carrier = amp * np.sin(2*np.pi*50*time)
noise = np.random.normal(scale=np.sqrt(noise_power),
                         size=time.shape)
x = carrier + noise

f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)
plt.figure(1)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')

Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
_, xrec = signal.istft(Zxx, fs)
plt.figure(2)
plt.plot(time, x, time, xrec, time, carrier)
plt.xlim([2, 2.1])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')
plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])

plt.figure(3)
plt.plot(time, x, time, xrec, time, carrier)
plt.xlim([0, 0.1])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')
plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])


plt.show()




