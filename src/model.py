import torch
import torch.nn as nn
from module import FeatureEmbedder
import yaml
from collections import Counter
import math
from preprocess import IndexCounter

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class Model(nn.Module):
    def __init__(self, data_config=None, emb_dim=8, hidden_dim=64, seq_len=128):
        super().__init__()
        self.ccba_emb = FeatureEmbedder(
            num_categorical=0,
            num_numerical=8,
            num_categories=0,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
        )
        self.cdtx_emb = FeatureEmbedder(
            num_categorical=2,
            num_numerical=1,
            num_categories=128 + 51,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
        )
        self.cust_emb = FeatureEmbedder(
            num_categorical=3,
            num_numerical=1,
            num_categories=4 + 22 + 11,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
        )
        self.dp_emb = FeatureEmbedder(
            num_categorical=8,
            num_numerical=2,
            num_categories=2 + 24 + 3 + 22 + 30 + 350 + 2 + 2,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
        )
        self.remit_emb = FeatureEmbedder(
            num_categorical=1,
            num_numerical=1,
            num_categories=5,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
        )

        self.net = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            # proj_size=1,
        )
        
        self.hidden_dim = hidden_dim
        self.positional_emb = PositionalEncoding(d_model=self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8,batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3)
        self.predictor = nn.Linear(self.hidden_dim, 1)
    def forward(self, events, orders, length, batch_size):
         
        feats = {}
        feats["ccba"] = self.ccba_emb(events["ccba"])
        feats["cdtx"] = self.cdtx_emb(events["cdtx"])
        feats["custinfo"] = self.cust_emb(events["custinfo"])
        feats["dp"] = self.dp_emb(events["dp"])
        feats["remit"] = self.remit_emb(events["remit"])
        feat = torch.empty(
            (batch_size, length, self.hidden_dim), device=feats["ccba"].device
        )
        while torch.isnan(feat).any():
            feat = torch.empty(
                (batch_size, length, self.hidden_dim), device=feats["ccba"].device
            )
        for key, value in orders.items():
            feat[value[0], value[1]] = feats[key]
        feat = self.positional_emb(feat)
        # feat = self.encoder(feat.contiguous())
        # return torch.sigmoid(self.predictor(feat)).clamp(min=1e-8, max=1-1e-8).squeeze(-1)
        return self.net(feat)[0].mean(dim=2)
if __name__ == "__main__":
    from dataset import TrainBaseDataset, TrainDataset
    from pathlib import Path
    from torch.utils.data import DataLoader
    import numpy as np
    model = Model().eval().cuda()
    meta_dataset = TrainBaseDataset(Path("data"))
    dataset = TrainDataset(
        meta_dataset, "data/train_x_alert_date.csv", "data/train_y_answer.csv"
    )

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
    for value in events.values():
        for key in value:
            value[key] = value[key].cuda()
    
    out = model(events, orders, max_seq, batch_size)
    out = out[range(len(targets)), targets]
    print(np.shape(targets))
    print(out.shape)
    print(np.shape(labels))
