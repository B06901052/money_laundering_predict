import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from utils import IndexCounter

def load_tables(data_path):
    """
    load each table, merge the date information into custinfo
    """
    """load_tables
    
    return 5 tables: ccba, cdtx, custinfo, dp, remit

    Returns:
        dict[str, DataFrame]: table name map to table
    """
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
    tableinfos = {}
    # iterate each table
    for table_name, table in tables.items():
        # accumulate the index for the categories in the same table
        idx_counter = IndexCounter(1)
        if table_name == "ccba":
            table["ratam"] = table.usgam / table.cycam
        for col_name in table:
            cfg = data_config[table_name][col_name]
            if cfg['type'] == "date":
                table[col_name] = table[col_name] / 365
            elif cfg['type'] == "categorical":
                # handle the missing value
                series = table[col_name].fillna(-1)
                # map the category to new label
                labels = sorted(series.unique())
                mapping[col_name] = dict(zip(labels, idx_counter))
                table[col_name] = series.apply(lambda x: mapping[col_name][x])
            elif cfg['type'] == "numerical":
                if col_name in {"lupay", "cycam", "usgam", "clamt", "csamt", "inamt", "cucsm", "cucah", "amt", "total_asset", "tx_amt", "trade_amount_usd"}:
                    table["log"+col_name] = table[col_name] - table[col_name].min()
                    table["log"+col_name].fillna(0, inplace=True)
                    table["log"+col_name] = np.log(table["log"+col_name]+1)
                    table["log"+col_name] = (table["log"+col_name] - table["log"+col_name].mean()) / table["log"+col_name].std()
                # normalization
                table[col_name] = (table[col_name] - table[col_name].mean()) / table[col_name].std()
                # handle the missing value
                table[col_name].fillna(0, inplace=True)
            elif cfg['type'] == "label":
                pass
            else:
                raise NotImplementedError
        tableinfos[table_name] = table.describe()
        tableinfos[table_name].loc["count"] /= len(table.cust_id.unique())
        tables[table_name] = table.groupby("cust_id")
    
    # save the mapping between original label and the index for embedding
    if (data_path / "mapping.pkl").exists():
        with open(data_path / "mapping.pkl", 'rb') as f:
            assert mapping == pickle.load(f), "mapping is different"
    else:
        with open(data_path / "mapping.pkl", 'wb') as f:
            pickle.dump(mapping, f)
        
    return tables, tableinfos
        
def concat_data(grouped_tables, tableinfos, data_config):
    # construct a dict which keys are all cust_ids
    data = {key: {} for key in grouped_tables["custinfo"].groups.keys()}
    for table_name, table in grouped_tables.items():
        data_type_counter = Counter(v['type'] for v in data_config[table_name].values())
        for cust_id, df in table:
            data[cust_id][table_name + "_summary"] = (df.describe() / tableinfos[table_name]).to_numpy().flatten()[len(tableinfos[table_name].columns)-1:]
            num = len(df)
            data[cust_id][table_name] = {}
            cat_idx_counter = IndexCounter()
            num_idx_counter = IndexCounter()
            categorical = np.full((num, data_type_counter["categorical"]), float('nan'), dtype=np.int64)
            numerical = np.full((num, data_type_counter["numerical"]), float('nan'), dtype=np.float32)
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

            assert (
                not np.isnan(date).any() and
                not np.isnan(categorical).any() and
                not np.isnan(numerical).any()
            ), "something is nan"

            data[cust_id][table_name]["date"] = date
            data[cust_id][table_name]["cat"] = categorical
            data[cust_id][table_name]["num"] = numerical
    
    # sort each event by date and record the event index
    for cust_id, tables in list(data.items()):
        event_index = []
        for table_name, table in tables.items():
            if not table_name.endswith("_summary"):
                event_index.extend((table_name, i, date) for i, date in enumerate(table["date"]))
        event_index.sort(key=lambda x: x[-1])
        data[cust_id]["event_index"] = event_index
    
    return data
    
    
if __name__ == "__main__":
    data_path = Path("data")
    with open(data_path / "data_config.yaml", 'r') as f:
        data_config = yaml.load(f, yaml.Loader)

    tables = load_tables(data_path)
    grouped_tables, tableinfos = data_preprocess(tables, data_config, data_path)
    data = concat_data(grouped_tables, tableinfos, data_config)
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
        