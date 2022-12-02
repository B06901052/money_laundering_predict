import pandas as pd
from itertools import chain
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import cpu_count
import random
import torch
import torch.nn as nn
# from pytorch_ranger import Ranger
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from model import Model
from dataset import TrainBaseDataset, TrainDataset
import argparse
import matplotlib.pyplot as plt
import numpy as np
def fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main(args):
    torch.autograd.set_detect_anomaly(True)
     # Do not change this
    fix(args.seed)
    model = Model().eval()
    meta_dataset = TrainBaseDataset(Path("data"))
    dataset = TrainDataset(
        meta_dataset,
        "data/train_x_alert_date.csv",
        "data/train_y_answer.csv",
        max_seq=128,
    )
    train_size = round(len(dataset) * 0.8)
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(1337),
    )
    test_dataset = TrainDataset(meta_dataset, "data/public_x_alert_date.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            torch.where(dataset.y[train_dataset.indices] == 1, 40, 1),
            num_samples=len(train_dataset),
        ),
        collate_fn=dataset.collate_fn,
        num_workers=cpu_count(),
        pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=cpu_count(),
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=cpu_count(),
    )

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    model = model.to(device)

    pbar = tqdm(range(args.epochs))
    train_loss, train_recall, valid_loss, valid_recall = [], [], [], []
    for epoch in pbar:
        total_loss = 0
        # training
        model.train()
        all_outs = []
        all_labels = []
        for events, orders, max_seq, targets, labels in tqdm(train_loader):
            for value in events.values():
                for key in value:
                    value[key] = value[key].to(device)
            all_labels.extend(labels.cpu().tolist())
            labels = labels.to(device=device, dtype=torch.float32)
            out = model(events, orders, max_seq, args.batch_size)[
                range(len(targets)), targets
            ]
            out = torch.clamp(out, min=1e-8, max=1-1e-8)
            loss = criterion(out, labels)
            if torch.isnan(loss).any():
                continue
            # assert not torch.isnan(loss).any()
            optim.zero_grad()
            loss.backward()
            all_outs.extend(out.tolist())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()
            # pbar.set_postfix({"loss": f"{loss.item():4.2f}"})
            total_loss += loss.item() * len(labels)
        total_loss /= len(train_loader.dataset)
        all_outs = enumerate(
            sorted(zip(all_outs, all_labels), key=lambda x: x[0], reverse=True)
        )
        all_outs = list(filter(lambda x: x[1][1] == 1, all_outs))
        total_loss /= len(valid_loader.dataset)
        
        recall = (len(all_outs) - 1) / (all_outs[-2][0] + 1)
        train_loss.append(total_loss)
        train_recall.append(recall)
        pbar.set_description(f"Train loss : {total_loss:.4f}, Recall : {recall:.4f}")
        pbar.refresh()
        model.eval()
        total_loss = 0
        all_outs = []
        all_labels = []
        with torch.no_grad():
            for events, orders, max_seq, targets, labels in tqdm(valid_loader):
                for value in events.values():
                    for key in value:
                        value[key] = value[key].to(device)
                all_labels.extend(labels.cpu().tolist())
                labels = labels.to(device=device, dtype=torch.float32)
                out = model(events, orders, max_seq, args.batch_size)[
                    range(len(targets)), targets
                ]
                out = torch.clamp(out, min=1e-8, max=1-1e-8)
                
                loss = criterion(out, labels)
                all_outs.extend(out.tolist())
                total_loss += loss.item() * len(labels)
            all_outs = enumerate(
                sorted(zip(all_outs, all_labels), key=lambda x: x[0], reverse=True)
            )
            all_outs = list(filter(lambda x: x[1][1] == 1, all_outs))
            total_loss /= len(valid_loader.dataset)
            
            recall = (len(all_outs) - 1) / (all_outs[-2][0] + 1)
            pbar.set_description(f"valid loss : {total_loss:.4f}, Recall : {recall:.4f}")
            pbar.refresh()
            valid_loss.append(total_loss)
            valid_recall.append(recall)
        plt.plot(train_loss, label='train loss')
        plt.plot(valid_loss, label='valid loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig('loss.jpg')
        plt.clf()
        plt.plot(train_recall, label='train recall')
        plt.plot(valid_recall, label='valid recall')
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.legend(loc='best')
        plt.savefig('recall.jpg')
        plt.clf()
    all_alert_keys = set(pd.read_csv("data/submission_sample.csv").alert_key)
    predict_alert_keys = []
    probs = []
    for events, orders, max_seq, targets, alert_keys in tqdm(test_loader):
        predict_alert_keys.extend(alert_keys)
        for value in events.values():
            for key in value:
                value[key] = value[key].to(device)
        out = model(events, orders, max_seq, args.batch_size)[
            range(len(targets)), targets
        ]
        out = torch.sigmoid(out)
        probs.extend(out.cpu().tolist())

    other_alert_keys = all_alert_keys - set(predict_alert_keys)

    lines = sorted(zip(predict_alert_keys, probs), key=lambda x: x[1], reverse=True)
    lines = "\n".join(
        chain(
            ["alert_key,probability"],
            map(lambda x: f"{x[0]},{x[1]}", lines),
            map(lambda x: f"{x},0.0", other_alert_keys),
            [""],
        )
    )
    with open("prediction.csv", "w") as f:
        f.write(lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # randomness
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--batch_size", type=int, default=256*2)
    parser.add_argument("--lr", type=float, default=1e-5)  
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50) 

    args = parser.parse_args()
    main(args)

