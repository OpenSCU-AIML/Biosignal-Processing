class TrialMetadata:
    def __init__(self, raw_metadata):
        self.raw = raw_metadata
        self.header = self._get_header()

    def _get_header(self):
        temp = [i for i in self.raw.loc[1, 0:2]]
        return {"app": temp[0], "datetime": temp[1], "duration": temp[2]}