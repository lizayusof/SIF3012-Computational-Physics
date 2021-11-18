#This code is taken from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
#This code Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated around 3kHz, corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.

import numpy as np
import matplotlib.pyplot as mpl
from scipy import signal


fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power),
                         size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

f, t, Zxx = signal.stft(x, fs, nperseg=1000)
mpl.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
mpl.title('STFT Magnitude')
mpl.ylabel('Frequency [Hz]')
mpl.xlabel('Time [sec]')
mpl.show()
