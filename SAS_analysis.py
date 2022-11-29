import pickle
from pathlib import Path
from pdb import set_trace

data_path = Path('data')

with open(data_path / 'data.pkl', 'rb') as f:
    data = pickle.load(f)

set_trace()