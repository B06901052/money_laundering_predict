import pandas as pd
from itertools import chain
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from pytorch_ranger import Ranger
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from model import Model
from dataset import TrainBaseDataset, TrainDataset


if __name__ == "__main__":
    model = Model().eval()
    meta_dataset = TrainBaseDataset(Path("data"))
    dataset = TrainDataset(meta_dataset, 'data/train_x_alert_date.csv', "data/train_y_answer.csv", max_seq=128)
    train_size = round(len(dataset) * 0.8)
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, len(dataset)-train_size],
        generator=torch.Generator().manual_seed(1337)
    )
    test_dataset = TrainDataset(meta_dataset, 'data/public_x_alert_date.csv')
    
    batch_size = 256
    epochs = 10
    lr = 1e-2
    weight_decay = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(torch.where(dataset.y[train_dataset.indices]==1, 40, 1), num_samples=len(train_dataset)),
        collate_fn=dataset.collate_fn,
        num_workers=cpu_count(),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=cpu_count(),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=cpu_count(),
    )
    
    optim = Ranger(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        total_loss = 0
        # training
        model.train()
        for events, orders, max_seq, targets, labels in tqdm(train_loader):
            for value in events.values():
                for key in value:
                    value[key] = value[key].to(device)
            labels = labels.to(device=device, dtype=torch.float32)
            out = model(events, orders, max_seq, batch_size)[range(len(targets)), targets]
            loss = criterion(out, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            pbar.set_postfix({"loss": f"{loss.item():4.2f}"})
            total_loss += loss.item() * len(labels)
        total_loss /= len(train_loader.dataset)
            
            
        model.eval()
        total_loss = 0
        all_outs = []
        all_labels = []
        for events, orders, max_seq, targets, labels in tqdm(valid_loader):
            for value in events.values():
                for key in value:
                    value[key] = value[key].to(device)
            all_labels.extend(labels.cpu().tolist())
            labels = labels.to(device=device, dtype=torch.float32)
            out = model(events, orders, max_seq, batch_size)[range(len(targets)), targets]
            loss = criterion(out, labels)
            all_outs.extend(out.tolist())
            total_loss += loss.item() * len(labels)
        all_outs = enumerate(sorted(zip(all_outs, all_labels), key=lambda x: x[0], reverse=True))
        all_outs = list(filter(lambda x: x[1][1]==1, all_outs))
        total_loss /= len(valid_loader.dataset)
        print("\n", total_loss, " ", (len(all_outs) - 1) / all_outs[-2][0])
        
    all_alert_keys = set(pd.read_csv("data/submission_sample.csv").alert_key)
    predict_alert_keys = []
    probs = []
    for events, orders, max_seq, targets, alert_keys in tqdm(test_loader):
        predict_alert_keys.extend(alert_keys)
        for value in events.values():
            for key in value:
                value[key] = value[key].to(device)
        out = model(events, orders, max_seq, batch_size)[range(len(targets)), targets]
        out = torch.sigmoid(out)
        probs.extend(out.cpu().tolist())
    
    other_alert_keys = all_alert_keys - set(predict_alert_keys)
    
    lines = sorted(zip(predict_alert_keys, probs), key=lambda x: x[1], reverse=True)
    lines = "\n".join(chain(
        ["alert_key,probability"],
        map(lambda x: f"{x[0]},{x[1]}", lines),
        map(lambda x: f"{x},0.0", other_alert_keys),
        [""]
    ))
    with open("prediction.csv", 'w') as f:
        f.write(lines)
        
        
    
        
    
            
    
        