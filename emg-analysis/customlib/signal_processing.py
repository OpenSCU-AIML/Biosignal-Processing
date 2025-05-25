import pandas as pd
from .data_structures import Sensor

class Sensors:
    def __init__(self, path, raw_data, verbose=False):
        self.path = path
        self.raw_data = raw_data
        self.columns = self._get_columns()
        self.sensors = self._get_sensors()
        self._load_sensors()
        self.verbose = verbose
        if verbose:
            print(f"Found {len(self.columns)-1} sensors at {self.columns}")

    def _get_columns(self):
        dd = self.raw_data.loc[0, 0::2]
        cols = []
        for i, j in enumerate(dd):
            if int(j.strip().split(" ")[1]) == 1:
                cols.append(i * 2)
        return cols

    def _get_sensors(self):
        sa = self.raw_data.loc[1, :]
        da = pd.read_csv(self.path, usecols=self.columns, skiprows=3, header=None, keep_default_na=False)
        slist = []
        next = 1
        tcol = self.columns + [40]
        for i in da:
            slist.append(
                Sensor(
                    i,
                    tcol[next],
                    da.loc[0, i],
                    da.loc[1, i],
                    self.raw_data.loc[1, i + 1],
                    da.loc[2, i],
                )
            )
            next += 1
        return slist

    def _load_sensors(self):
        chunks = [self.raw_data.loc[:, sen.idx:sen.nchan-1] for sen in self.sensors]
        for n, chunk in enumerate(chunks):
            self.sensors[n].load_channels(chunk)

    @property
    def channel_metadata(self):
        return self.raw_data.loc[0:1, 1::2]