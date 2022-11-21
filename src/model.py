import torch
import torch.nn as nn
from module import FeatureEmbedder
import yaml
from collections import Counter

from preprocess import IndexCounter

class Model(nn.Module):
    def __init__(self, data_config=None, emb_dim=8, hidden_dim=64):
        super().__init__()
        self.ccba_emb  = FeatureEmbedder(0, 8, 0, emb_dim, hidden_dim)
        self.cdtx_emb  = FeatureEmbedder(2, 1, 128+51, emb_dim, hidden_dim)
        self.cust_emb  = FeatureEmbedder(3, 1, 4+22+11, emb_dim, hidden_dim)
        self.dp_emb    = FeatureEmbedder(8, 2, 2+24+3+22+30+350+2+2, emb_dim, hidden_dim)
        self.remit_emb = FeatureEmbedder(1, 1, 5, emb_dim, hidden_dim)
        
        self.net = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            proj_size=1,
        )
        
        self.hidden_dim = hidden_dim
        
    def forward(self, events, orders, length, batch_size):
        feats = {}
        feats["ccba"] = self.ccba_emb(events["ccba"])
        feats["cdtx"] = self.cdtx_emb(events["cdtx"])
        feats["custinfo"] = self.cust_emb(events["custinfo"])
        feats["dp"] = self.dp_emb(events["dp"])
        feats["remit"] = self.remit_emb(events["remit"])
        feat = torch.empty((batch_size, length, self.hidden_dim), device=feats["ccba"].device)
        for key, value in orders.items():
            feat[value[0], value[1]] = feats[key]
        return self.net(feat)[0].mean(dim=2)
        
if __name__ == "__main__":
    from dataset import TrainBaseDataset, TrainDataset
    from pathlib import Path
    from torch.utils.data import DataLoader

    model = Model().eval()
    meta_dataset = TrainBaseDataset(Path("data"))
    dataset = TrainDataset(meta_dataset, 'data/train_x_alert_date.csv', "data/train_y_answer.csv")
    
    batch_size = 2
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=2,
    )
    loader = iter(loader)
    batch = next(loader)
    events, orders, max_seq, targets, labels = batch
    out = model(events, orders, max_seq, batch_size)
    out = out[range(len(targets)), targets]
    print(targets)
    print(out)
    print(labels)