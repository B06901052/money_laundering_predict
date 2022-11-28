import torch
import torch.nn as nn
from module import FeatureEmbedder, SingleEventPredictor

class Model(nn.Module):
    def __init__(self, data_config=None, emb_dim=8, hidden_dim=64):
        super().__init__()
        self.ccba_emb  = FeatureEmbedder(0, 8, 0, emb_dim, hidden_dim)
        self.cdtx_emb  = FeatureEmbedder(2, 1, 128+51, emb_dim, hidden_dim)
        self.cust_emb  = FeatureEmbedder(3, 1, 4+22+11, emb_dim, hidden_dim)
        self.dp_emb    = FeatureEmbedder(8, 2, 2+24+3+22+30+350+2+2, emb_dim, hidden_dim)
        self.remit_emb = FeatureEmbedder(1, 1, 5, emb_dim, hidden_dim)
        
        self.ccba_pred  = SingleEventPredictor(8, [0], hidden_dim)
        self.cdtx_pred  = SingleEventPredictor(1, [128,51], hidden_dim)
        self.cust_pred  = SingleEventPredictor(1, [4,22,11], hidden_dim)
        self.dp_pred    = SingleEventPredictor(2, [2,24,3,22,30,350,2,2], hidden_dim)
        self.remit_pred = SingleEventPredictor(1, [5], hidden_dim)
        
        self.pred_crit = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]))
        
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=4),
        )
        self.net = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        # trlayer = nn.TransformerEncoderLayer(
        #     hidden_dim,
        #     nhead=8,
        #     dim_feedforward=1024,
        #     dropout=0.1,
        #     activation=nn.GELU(),
        #     batch_first=True,
        #     norm_first=True,
        # )
        # self.net = nn.TransformerEncoder(
        #     trlayer,
        #     num_layers=4,
        #     norm = nn.LayerNorm(hidden_dim),
        # )
        # self.postnet = nn.TransformerEncoder(
        #     trlayer,
        #     num_layers=2,
        #     norm = nn.LayerNorm(hidden_dim),
        # )
        
        self.pred_head = nn.Sequential(
            # nn.Linear(hidden_dim<<1, hidden_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim<<1, 1),
        )
        
        self.hidden_dim = hidden_dim
        
    def forward(self, events, orders, length, batch_size, targets, labels=None):
        # embed the event
        feats = {}
        feats["ccba"] = self.ccba_emb(events["ccba"])
        feats["cdtx"] = self.cdtx_emb(events["cdtx"])
        feats["custinfo"] = self.cust_emb(events["custinfo"])
        feats["dp"] = self.dp_emb(events["dp"])
        feats["remit"] = self.remit_emb(events["remit"])
        # cat event into sequence
        feat = torch.empty((batch_size, length, self.hidden_dim), device=feats["ccba"].device)
        for key, value in orders.items():
            feat[value[0], value[1]] = feats[key].to(dtype=feat.dtype)
        # conv sequence
        feat = self.conv(feat.transpose(1,2)).transpose(1,2)
        # rnn sequence
        feat = self.net(feat)[0]
        # predict sar
        pred = self.pred_head(feat[range(len(targets)), targets]).squeeze(-1)
        if labels is not None:
            pred_loss = self.pred_crit(pred, labels)
        
            total_loss = pred_loss
            return pred, total_loss
        else:
            return pred
        
if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    from dataset import TrainMetaDataset, TrainDataset

    model = Model().eval()
    meta_dataset = TrainMetaDataset(Path("data"))
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