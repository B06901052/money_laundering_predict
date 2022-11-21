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
    
    batch_size = 128
    epochs = 10
    lr = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(torch.where(dataset.y[train_dataset.indices]==1, 20, 1), num_samples=len(train_dataset)),
        collate_fn=dataset.collate_fn,
        num_workers=cpu_count(),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=cpu_count(),
        pin_memory=True,
    )
    
    optim = Ranger(model.parameters(), lr=lr)
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
            pbar.set_postfix({"loss": loss.item(), "out": out[0].item()})
            total_loss += loss.item() * len(labels)
        total_loss /= len(train_loader.dataset)
            
            
        model.eval()
        total_loss = 0
        all_outs = []
        all_labels = []
        for events, orders, max_seq, targets, labels in tqdm(train_loader):
            for value in events.values():
                for key in value:
                    value[key] = value[key].to(device)
            labels = labels.to(device=device, dtype=torch.float32)
            out = model(events, orders, max_seq, batch_size)[range(len(targets)), targets]
            loss = criterion(out, labels)
            all_outs.extend(out)
            total_loss += loss.item() * len(labels)
        total_loss /= len(valid_loader.dataset)
        print("\n", total_loss)
            
            
        