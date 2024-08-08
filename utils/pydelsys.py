import os
import pandas as pd
import os

class DS:
    def __init__(self, root, path, trials=None):
        self.root = root
        self.path = path
        self.trials = [f"{root}/{path}/{i}" for i in os.listdir(f"{root}/{path}")]

class TMeta:
    def __init__(self, raw, header=None):
        self.raw = raw
        self.header = self.getHeader(raw)

    def getHeader(self, raw):
        temp = [i for i in raw.loc[1, 0:2]]
        return {"app": temp[0], "datetime": temp[1], "duration": temp[2]}
class Trial:
    def __init__(self, path, meta=None, rawban=None, rawchan=None):
        self.path = path
        self.rawban = (
            pd.read_csv(self.path, usecols=[0, 1], header=None)
            .loc[0:6, 0:4]
            .astype("string")
            .T
        )
        self.rawchan = pd.read_csv(
            self.path, skiprows=5, header=None, keep_default_na=False
        )
        self.meta = TMeta(self.rawban)


class Chan:
    def __init__(self, id, rc, T=None, V=None):
        self.id = id
        self.rc = rc

    def getDf(self):
        self.rc.columns = self.rc.iloc[0]
        self.rc = self.rc[2:]
        self.rc = self.rc.astype("float64")
    def R(self):
        return self.rc
    def T(self):
        return self.rc.iloc[:,0].to_numpy()
    def V(self):
        return self.rc.iloc[:,-1].to_numpy()
class Sensor:
    def __init__(self, idx, nchan, name, st, sr, mode, chans=None):
        self.idx = idx
        self.nchan = nchan - idx
        self.name = name.split(":")[-1].strip()
        self.st = st
        self.sr = sr
        self.mode = mode.strip().split(" ")[0]
        self.chans = chans

    def info(self):
        return f" {self.name} {self.nchan//2}x channels @ {self.sr}"

    def getChans(self, rchunk):
        self.chans = [
            Chan(n//2, rchunk.loc[:, n : n + 1]) for n in range(0, self.nchan, 2)
        ]
        print(f" {self.name} {self.nchan//2}x channels @ {self.sr}")

    def setChans(self):
        for ch in self.chans:
            ch.getDf()
    def npChans(self):
        return [ch.V() for ch in self.chans]

class Sensors:
    def __init__(self, path, rc, v=False, cols=None, sens=None):
        self.path = path
        self.rc = rc
        self.cols = self.getRcol(rc)
        self.sens = self.getSens()
        self.loadSens()
        self.v = v
        if v:
            print(f"Found {len(self.cols)-1} sensors at {self.cols}")

    def getRcol(self, rc):  # get chan widths
        dd = rc.loc[0, 0::2]
        cols = []
        for i, j in enumerate(dd):
            if int(j.strip().split(" ")[1]) == 1:
                cols.append(i * 2)
                rc.loc[1, (i * 2)]
        return cols

    def getSens(self):
        sa = self.rc.loc[1, :]
        da = pd.read_csv(
            self.path, usecols=self.cols, skiprows=3, header=None, keep_default_na=False
        )
        slist = []
        next = 1
        tcol = self.cols
        tcol.append(40)
        for i in da:
            slist.append(
                Sensor(
                    i,
                    tcol[next],
                    da.loc[0, i],
                    da.loc[1, i],
                    self.rc.loc[1, i + 1],
                    da.loc[2, i],
                )
            )
            next = next + 1
        return slist
    def loadSens(self):
        chunks = [self.rc.loc[:, sen.idx : sen.nchan - 1] for sen in self.sens]
        for n in range(len(chunks)):
            self.sens[n].getChans(chunks[n])
            self.sens[n].setChans()
    def getCmeta(self):
        return self.rc.loc[0:1, 1::2]