class IndexCounter:
    def __init__(self, start_idx=0):
        self.idx = start_idx - 1
        
    def __iter__(self):
        while True:
            self.idx += 1
            yield self.idx
    def __next__(self):
        self.idx += 1
        return self.idx