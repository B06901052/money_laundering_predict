import pickle

import torch
from torch.utils.data import Dataset
from pdb import set_trace
from preprocess import IndexCounter

class PretrainDataset:
    def __init__(self, data_path, max_len=64):
        with open(data_path / 'data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        idx_counter = IndexCounter()
        self.cust_id_map = {key: next(idx_counter) for key in self.data}
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cust_id = self.cust_id_map[idx]
        return self.data[cust_id]
    
    def collate_fn(self, batch):
        max_len = min(self.max_len, max(len(sample["event_index"]) for sample in batch))
        for sample in batch:
            # TODO
            pass
            
class TrainBaseDataset:
    def __init__(self, data_path):
        with open(data_path / 'data.pkl', 'rb') as f:
            self.data = pickle.load(f)
            
        self.alert = {}
        set_trace()
        for cust_id, value in self.data.items():
            tmp = enumerate(value["event_index"])
            for event_idx, (_, custinfo_idx, _) in filter(lambda x: x[1][0] == "custinfo", tmp):
                alert_key = value["custinfo"]["alert_key"][custinfo_idx]
                self.alert[alert_key] = (cust_id, event_idx)
    
    def __len__(self):
        return len(self.alert)
    
    def __getitem__(self, alert_key):
        cust_id, event_idx = self.alert[alert_key]
        return self.data[cust_id], event_idx

class TrainDataset(Dataset):
    def __init__(self, train_base_dataset, x_path, y_path=None, max_seq=128):
        self.train_base_dataset = train_base_dataset
        self.max_seq = max_seq
        if y_path:
            with open(y_path, 'r') as f:
                # alert_key: sar_flag
                tmp = dict(tuple(line.split(",")) for line in f.read().split("\n")[1:-1])
                self.x = list(tmp.keys())
                self.y = torch.LongTensor(list(map(lambda x: int(x), tmp.values())))
        else:
            with open(x_path, 'r') as f:
                # alert_key: sar_flag
                self.x = [line.split(",")[0] for line in f.read().split("\n")[1:-1]]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        alert_key = int(self.x[idx])
        if hasattr(self, "y"):
            return self.train_base_dataset[alert_key], self.y[idx]
        else:
            return self.train_base_dataset[alert_key], alert_key
    
    def collate_fn(self, batch):
        batch, label = zip(*batch)
        max_seq = max(len(sample[0]["event_index"]) for sample in batch)
        max_seq = min(max_seq, self.max_seq)
        targets = []
        events = {
            key: {k: [] for k in ["num", "cat", "date"]}
                for key in ["ccba", "cdtx", 'custinfo', 'dp', 'remit']
        }
        orders = {key: [[],[]] for key in ["ccba", "cdtx", 'custinfo', 'dp', 'remit']}
        for batch_id, (data, alert_event_idx) in enumerate(batch):
            idx_counter = IndexCounter()
            length = len(data['event_index'])
            s = alert_event_idx - max_seq // 2
            e = s + max_seq
            if s < 0:
                s = 0
                e = max_seq
            elif e > length:
                s = length - max_seq
                e = length
            for event in data['event_index'][s:e]:
                event_type, idx, _ = event
                events[event_type]['num'].append(torch.from_numpy(data[event_type]['num'][idx]))
                events[event_type]['cat'].append(torch.from_numpy(data[event_type]['cat'][idx]))
                events[event_type]['date'].append(data[event_type]['date'][idx])
                orders[event_type][0].append(batch_id)
                orders[event_type][1].append(next(idx_counter))
                
            targets.append(alert_event_idx - s)
        for value in events.values():
            if value['num']:
                value['num'] = torch.stack(value['num'])
            else:
                value['num'] = torch.empty((0,0), dtype=torch.float32)
            if value['cat']:
                value['cat'] = torch.stack(value['cat'])
            else:
                value['cat'] = torch.empty((0,0), dtype=torch.int64)
            value['date'] = torch.Tensor(value['date'])
        
        if hasattr(self, "y"):
            return events, orders, max_seq, targets, torch.LongTensor(label)
        else:
            return events, orders, max_seq, targets, label
    
if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from multiprocessing import cpu_count
    meta_dataset = TrainBaseDataset(Path("data"))
    dataset = TrainDataset(meta_dataset, 'data/train_x_alert_date.csv', "data/train_y_answer.csv")
    # loader = DataLoader(
    #     dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     collate_fn=dataset.collate_fn,
    #     num_workers=cpu_count(),
    # )
    # # check all batchs are fine
    # for _ in tqdm(loader):
    #     pass
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=2,
    )
    loader = iter(loader)
    batch = next(loader)
    events, orders, max_seq, targets, label = batch
    print(orders)
    print(events.keys())
    for key in events['ccba']:
        print(key)
        print(events['ccba'][key])
        print(events['ccba'][key].shape)
    print("empty feature will be: \n", events['remit'])
    print(orders.keys())
    print(max_seq)
    print(targets)