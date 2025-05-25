import os
import pandas as pd
import numpy as np

class DataSet:
    def __init__(self, root, path):
        self.root = root
        self.path = path
        self.trials = [f"{root}/{path}/{i}" for i in os.listdir(f"{root}/{path}")]

class Trial:
    def __init__(self, path):
        self.path = path
        self.rawban = pd.read_csv(self.path, usecols=[0, 1], header=None).loc[0:6, 0:4].astype("string").T
        self.rawchan = pd.read_csv(self.path, skiprows=5, header=None, keep_default_na=False)
        self.meta = None
        self.sensors = None

    def load_metadata(self):
        from .metadata import TrialMetadata
        self.meta = TrialMetadata(self.rawban)

    def load_sensors(self):
        from .signal_processing import Sensors
        self.sensors = Sensors(self.path, self.rawchan)

class Channel:
    def __init__(self, id, raw_data):
        self.id = id
        self.raw_data = raw_data
        self.data = None

    def process_data(self):
        self.data = self.raw_data.copy()
        self.data.columns = self.data.iloc[0]
        self.data = self.data[2:].astype("float64")

    @property
    def time(self):
        return self.data.iloc[:, 0].to_numpy()

    @property
    def voltage(self):
        return self.data.iloc[:, -1].to_numpy()

class Sensor:
    def __init__(self, idx, nchan, name, st, sr, mode):
        self.idx = idx
        self.nchan = nchan - idx
        self.name = name.split(":")[-1].strip()
        self.st = st
        self.sr = sr
        self.mode = mode.strip().split(" ")[0]
        self.channels = []

    def __str__(self):
        return f"{self.name} {self.nchan//2}x channels @ {self.sr}"

    def load_channels(self, raw_chunk):
        self.channels = [Channel(n//2, raw_chunk.loc[:, n:n+1]) for n in range(0, self.nchan, 2)]
        for channel in self.channels:
            channel.process_data()

    @property
    def numpy_channels(self):
        return [ch.voltage for ch in self.channels]