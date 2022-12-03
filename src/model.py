from collections import Counter
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .module import FeatureEmbedder, SingleEventPredictor

class Model(nn.Module):
    def __init__(self, data_config=None, emb_dim=8, hidden_dim=128):
        super().__init__()
        self.event_emb = nn.ModuleDict({
            "ccba": FeatureEmbedder(0, 8+(1+8), 0, emb_dim, hidden_dim),
            "cdtx": FeatureEmbedder(2, 1+(1), 128+51, emb_dim, hidden_dim),
            "custinfo": FeatureEmbedder(3, 1+(1), 4+22+11, emb_dim, hidden_dim),
            "dp": FeatureEmbedder(8, 2+(1), 2+24+3+22+30+350+2+2, emb_dim, hidden_dim),
            "remit": FeatureEmbedder(1, 1+(1), 5, emb_dim, hidden_dim),
        })
        
        # self.event_pred  = nn.ModuleDict({            
        #     "ccba": SingleEventPredictor(8+(1+8), [0], hidden_dim),
        #     "cdtx": SingleEventPredictor(1+(1), [128,51], hidden_dim),
        #     "custinfo": SingleEventPredictor(1+(1), [4,22,11], hidden_dim),
        #     "dp": SingleEventPredictor(2+(1), [2,24,3,22,30,350,2,2], hidden_dim),
        #     "remit": SingleEventPredictor(1+(1), [5], hidden_dim),
        # })

        
        self.pred_crit = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]))
        
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=4),
            nn.BatchNorm1d(hidden_dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim>>1, 5, padding=4, groups=4, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim>>1, hidden_dim>>1, 5, padding=4, groups=4, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim>>1, hidden_dim>>1, 5, padding=4, groups=4, dilation=2),
        )
        trlayer = nn.TransformerEncoderLayer(
            hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim<<2,
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(
            trlayer,
            num_layers=2,
            norm = nn.LayerNorm(hidden_dim),
        )
        # self.postnet = nn.TransformerEncoder(
        #     trlayer,
        #     num_layers=2,
        #     norm = nn.LayerNorm(hidden_dim),
        # )
        
        self.summary_net = nn.Sequential(
            nn.Dropout(.5, True),
            nn.Linear(127+36+50+85+29, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False),
            nn.ReLU(True),
            weight_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim, affine=False),
        )
        
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim<<1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.hidden_dim = hidden_dim
        self.pad_embedding = nn.Parameter(torch.randn((1,1,hidden_dim)))
        
    def forward(self, events, orders, length, batch_size, targets, labels=None, summarys=None):
        # embed the event
        feats = {
            key: self.event_emb[key](event) for key, event in events.items()
        }
        # cat event into sequence
        feat = self.pad_embedding.repeat((batch_size, length, 1))
        mask = torch.ones(feat.shape[:2], dtype=torch.bool, device=feat.device)
        key_mask = torch.ones(feat.shape[:2], dtype=torch.bool, device=feat.device)
        for key, value in orders.items():
            if (
                self.training
            ):
                index = list(map(lambda x: .8 > random.random() or targets[value[0][x[0]]] == value[1][x[0]], enumerate(zip(*value))))

                batch_id, seq_id = [value[0][i] for i, b in enumerate(index) if b], [value[1][i] for i, b in enumerate(index) if b]
                feat[batch_id, seq_id] = feats[key][index].to(dtype=feat.dtype)
                mask[batch_id, seq_id] = False
            else:
                feat[value[0], value[1]] = feats[key].to(dtype=feat.dtype)
                mask[value[0], value[1]] = False
            key_mask[value[0], value[1]] = False

        # conv sequence
        L = feat.size(1)
        # feat2 = nn.functional.interpolate(self.conv2(feat.transpose(1,2)), L).transpose(1,2)
        # feat = torch.cat((self.conv(feat.transpose(1,2)).transpose(1,2), feat2), dim=2)
        feat = self.conv(feat.transpose(1,2)).transpose(1,2)
        # rnn sequence
        # feat = self.attn1(feat, feat, feat, key_padding_mask=mask)[0]
        feat = self.net(feat, src_key_padding_mask=key_mask)
        # predict sar
        summary_feat = self.summary_net(summarys.nan_to_num(posinf=0, neginf=0))
        pred = self.pred_head(torch.cat(((feat[range(len(targets)), targets]).squeeze(-1), summary_feat), dim=1)).squeeze(-1)
        # pred = self.pred_head((feat[range(len(targets)), targets]).squeeze(-1)).squeeze(-1)
        if labels is not None:
            # pred_loss = self.pred_crit(pred, labels)
            # TODO: label smoothing?
            pred_loss = self.pred_crit(pred, (labels + torch.randn_like(labels) * 0.1).clamp(0,1))
            # predict event
            # feat = self.postnet(feat, src_key_padding_mask=key_mask)
            # event_loss = 0
            # for key, value in orders.items():
            #     event_loss = event_loss + self.event_pred[key](
            #         feat[value[0], value[1]][mask[value[0], value[1]]],
            #         events[key]["num"][mask[value[0], value[1]]],
            #         events[key]["cat"][mask[value[0], value[1]]]
            #     )
            return pred, pred_loss, torch.zeros_like(pred_loss) #0.1 * event_loss
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