# %%
import os
from sys import path
path.append('../utils')
import pydelsys
# %%
dataset = pydelsys.DS(os.getcwd(), "naruto")
trial = pydelsys.Trial(dataset.trials[0])
sensors = pydelsys.Sensors(trial.path, trial.rawchan, True)
sensors.sens[0].info()

# %%
(L, R, U, D) = sensors.sens[0:4]

# %%
L.info()

# %%
L.chans[0].T()

# %%
L.chans[0].R()
# %%
import matplotlib.pyplot as plt
fig = plt.figure()
for ch in L.chans:
    print(ch.id)

# plt.plot(L.chans[0].T(),L.chans[0].V())

# %%
