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
# DISPLAY
vis.chanplot(channelArrays)


# %%
# hi and low bandpass filter
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# %%
# robust detection threshold
import numpy as np
# median abs dev
def threshMAD(arr):
    mean = np.mean(arr)
    dev = [np.absolute(x-mean)/0.6745 for x in arr]
    return np.median(dev)
def threshIdx(arr, thresh):
    return [idx for idx,x in enumerate(arr) if (x <= -thresh)]
def detectSpikes(y,Fs):
    th = threshMAD(y)
    threshIdx(y, thresh)


# %%
# LOAD 
nchan = sensors.sens[0].nchan//2
channels = channelArrays
# FILTER: Sample rate and desired cutoff frequencies (in Hz).
fs = float(sensors.sens[0].sr.strip().split(' ')[0])
lowcut = 20.0
highcut = 150.0 # 50% fmax
filtered = [butter_bandpass_filter(chan, lowcut, highcut, fs, order=3) for chan in channels]

# %%
def samplebytime(channels, s=0.1):
    T = s
    idx = int(5.8*fs)
    nsamples = int(T * fs) 
    t = np.linspace(0, T, nsamples, endpoint=False)
    cur = (idx,idx+nsamples)
    data = [channels[i][cur[0]:cur[1]] for i in range(nchan)]
    print(f"{T} seconds | samples: {nsamples}")
    return data
# %%
vis.chanplot(filtered)


# %% 
# THRESHOLD:
thresh = [threshMAD(chan) for chan in filtered]
spikes = [threshIdx(filtered[i],thresh[i]) for i in range(len(filtered))]


# %%
print(len(filtered))
# %%
vis.tickplot(filtered, spikes)
# %%
type(ticks[3][0])
# %%
sample = filtered[0][0:500]
# tt = threshIdx(sample,threshMAD(sample))
th = np.ones(len(sample)) * -1 * threshMAD(sample)
print(len(sample),len(th))
comp = np.c_[sample,th]
plt.plot(range(len(sample)),comp)

# %%
mad = MAD(data[0])
rmad = rMAD(data[0])
# %%
print(mad)
print(rmad)

# %%
