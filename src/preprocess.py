import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch

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

def load_tables(data_path):
    tables = {}
    tables["ccba"] = pd.read_csv(data_path / "public_train_x_ccba_full_hashed.csv")
    tables["cdtx"] = pd.read_csv(data_path / "public_train_x_cdtx0001_full_hashed.csv")
    tables["custinfo"] = pd.read_csv(data_path / "public_train_x_custinfo_full_hashed.csv")
    tables["dp"] = pd.read_csv(data_path / "public_train_x_dp_full_hashed.csv")
    tables["remit"] = pd.read_csv(data_path / "public_train_x_remit1_full_hashed.csv")
    public_x = pd.read_csv(data_path / "public_x_alert_date.csv")
    train_x = pd.read_csv(data_path / "train_x_alert_date.csv")
    # merge date to custinfo
    tables["custinfo"] = tables["custinfo"].merge(
        pd.concat((train_x, public_x), ignore_index=True),
        left_on="alert_key", right_on="alert_key"
    ).sort_values(by="alert_key")
    return tables



def data_preprocess(tables, data_config, data_path):
    mapping = {}
    for table_name, table in tables.items():
        idx_counter = IndexCounter()
        for col_name in table:
            cfg = data_config[table_name][col_name]
            if cfg['type'] == "date":
                table[col_name] = table[col_name] / 365
            elif cfg['type'] == "categorical":
                series = table[col_name].fillna(-1)
                labels = sorted(series.unique())
                mapping[col_name] = dict(zip(labels, idx_counter))
                table[col_name] = series.apply(lambda x: mapping[col_name][x])
            elif cfg['type'] == "numerical":
                table[col_name] = (table[col_name] - table[col_name].mean()) / table[col_name].std()
            elif cfg['type'] == "label":
                pass
            else:
                raise NotImplementedError
        tables[table_name] = table.groupby("cust_id")
    
    # save the mapping between original label and the index for embedding
    if (data_path / "mapping.pkl").exists():
        with open(data_path / "mapping.pkl", 'rb') as f:
            assert mapping == pickle.load(f), "mapping is different"
    else:
        with open(data_path / "mapping.pkl", 'wb') as f:
            pickle.dump(mapping, f)
        
    return tables
        
def concat_data(grouped_tables, data_config):
    data = {key: {} for key in grouped_tables["custinfo"].groups.keys()}
    for table_name, table in grouped_tables.items():
        data_type_counter = Counter(v['type'] for v in data_config[table_name].values())
        for cust_id, df in table:
            num = len(df)
            data[cust_id][table_name] = {}
            cat_idx_counter = IndexCounter()
            num_idx_counter = IndexCounter()
            categorical = np.empty((num, data_type_counter["categorical"]), dtype=np.int64)
            numerical = np.empty((num, data_type_counter["numerical"]), dtype=np.float32)
            for col_name in df:
                cfg = data_config[table_name][col_name]
                if cfg['type'] == "date":
                    date = np.array(df[col_name], dtype=np.float32)
                elif cfg['type'] == "categorical":
                    categorical[:, next(cat_idx_counter)] = df[col_name].to_numpy(dtype=np.int64)
                elif cfg['type'] == "numerical":
                    numerical[:, next(num_idx_counter)] = df[col_name].to_numpy(dtype=np.float32)
                elif cfg['type'] == "label":
                    data[cust_id][table_name][col_name] = df[col_name].tolist()
                else:
                    raise NotImplementedError

            data[cust_id][table_name]["date"] = date
            data[cust_id][table_name]["cat"] = categorical
            data[cust_id][table_name]["num"] = numerical
    
    for cust_id, tables in list(data.items()):
        event_index = []
        for table_name, table in tables.items():
            event_index.extend(((table_name, i, date) for i, date in enumerate(table["date"])))
        event_index.sort(key=lambda x: x[-1])
        data[cust_id]["event_index"] = event_index
    
    return data
    
    
if __name__ == "__main__":
    data_path = Path("data")
    with open(data_path / "data_config.yaml", 'r') as f:
        data_config = yaml.load(f, yaml.Loader)

    tables = load_tables(data_path)
    grouped_tables = data_preprocess(tables, data_config, data_path)
    data = concat_data(grouped_tables, data_config)
    with open(data_path / "data.pkl", 'wb') as f:
        pickle.dump(data, f)
    """
    The final data will be something like:
    {
        cust_id: {
            table_name: {
                date: np.ndarray (shape: (num, ))
                cat: np.ndarray(shape: (num, # of categorical features))
                num: np.ndarray(shape: (num, # of numerical features))
            }
            ...
            "event_index": [(table_name, index), ...]
        }
        ...
    }
    """
        