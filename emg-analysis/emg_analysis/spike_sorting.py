# %%
import os
from sys import path
path.append("../../utils")
import pydelsys
import matplotlib.pyplot as plt
import numpy as np
import visual
root = "/".join(os.getcwd().split("/")[:-2])
dataroot = f"{root}/datasets"
datadirs = [dir for dir in os.listdir(dataroot) if -1 == dir.find(".")]

#%%%
datadirs

#%%
dataset = pydelsys.DS(dataroot, "henry_data")
#%%
dataset.trials

#%%
trial = pydelsys.Trial(dataset.trials[0])
# %%
sensors = pydelsys.Sensors(trial.path, trial.rawchan, True)
sensors.sens[0].info()
channelArrays = sensors.sens[0].npChans()
vis = visual.ChanPlot(channelArrays)

# %%
vis.chanplot(channelArrays)


# %%

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# %%
fs = float(sensors.sens[0].sr.strip().split(' ')[0])
nchan = sensors.sens[0].nchan//2
channels = channelArrays
# %%
# Sample rate and desired cutoff frequencies (in Hz).
# fs = 2222.0
lowcut = 500.0
highcut = 1111.0
# %%
# Filter a noisy signal.
T = 0.1
idx = int(5.8*fs)
nsamples = int(T * fs) 
t = np.linspace(0, T, nsamples, endpoint=False) # Generate the time vector 
cur = (idx,idx+nsamples)
data = [channels[i][cur[0]:cur[1]] for i in range(nchan)]
for i in range(nchan):
    y = butter_bandpass_filter(data[i], lowcut, highcut, fs, order=6)
    plt.subplot(4, 1, i+1)
    plt.plot(t,y,label=f'ch {i+1}' )

plt.xlabel('time (seconds)')
plt.grid(True)
plt.legend(loc='upper left')

plt.show()

# %%
