import math
import random
import numpy as np
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
from dataset import TrainMetaDataset, TrainDataset

class Runner:
    def __init__(self, args, runner_config={}, model_config={}):
        self.args = args
        self.cfg = runner_config
        self._set_seed(self.args.seed, self.args.deterministic)
        self.model = Model().to(self.args.device)
        self.meta_dataset = TrainMetaDataset(self.args.data_path)
        self.datasets = self._get_dataset()
        self.loaders = {
            mode: self._get_dataloader(mode)
            for mode in self.args.split
        }
        
    def _set_seed(self, seed=1337, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
    
    def _get_optimizer(self):
        cfg = self.cfg["optimizer"]
        name = cfg.pop("name")
        return eval(name)(self.model.parameters(), **cfg)
    
    def _get_dataset(self):
        datasets = {}
        dataset = TrainDataset(
            train_base_dataset=self.meta_dataset,
            x_path='data/train_x_alert_date.csv',
            y_path="data/train_y_answer.csv",
            max_seq=self.cfg['datarc']["max_seq"],
            min_seq_ratio=.5,
            istrain=True
        )
        train_size = round(len(dataset) * self.cfg['datarc']["train_ratio"])
        datasets["train"], _ = random_split(
            dataset,
            [train_size, len(dataset)-train_size],
            generator=torch.Generator().manual_seed(self.args.seed)
        )
        dataset = TrainDataset(
            train_base_dataset=self.meta_dataset,
            x_path='data/train_x_alert_date.csv',
            y_path="data/train_y_answer.csv",
            max_seq=self.cfg['datarc']["max_seq"],
        )
        _, datasets["valid"] = random_split(
            dataset,
            [train_size, len(dataset)-train_size],
            generator=torch.Generator().manual_seed(self.args.seed)
        )
        datasets['test'] = TrainDataset(
            train_base_dataset=self.meta_dataset,
            x_path='data/public_x_alert_date.csv',
            max_seq=self.cfg['datarc']["max_seq"],
        )
        
        return datasets
    
    def _get_dataloader(self, mode):
        if mode == "train":
            return DataLoader(
                self.datasets["train"],
                batch_size=self.cfg["datarc"]["batch_size"],
                sampler=WeightedRandomSampler(torch.where(self.datasets["train"].dataset.y[self.datasets["train"].indices]==1, 40, 1), num_samples=self.cfg['datarc']['batch_size']*100),
                collate_fn=self.datasets["train"].dataset.collate_fn,
                num_workers=self.cfg["datarc"]["num_workers"],
                pin_memory=True,
            )
        elif mode == "valid":
            return DataLoader(
                self.datasets["valid"],
                batch_size=self.cfg["datarc"]["batch_size"],
                shuffle=False,
                collate_fn=self.datasets["valid"].dataset.collate_fn,
                num_workers=self.cfg["datarc"]["num_workers"],
                pin_memory=True,
            )
        elif mode == "test":
            return DataLoader(
                self.datasets["test"],
                batch_size=self.cfg["datarc"]["batch_size"],
                shuffle=False,
                collate_fn=self.datasets["test"].collate_fn,
                num_workers=self.cfg["datarc"]["num_workers"],
            )
    
    def train(self):
        self.optim = self._get_optimizer()
        pbar = tqdm(range(self.cfg["runner"]["n_epochs"]))
        for _ in pbar:
            total_loss = 0
            # training
            self.model.train()
            for batch_id, (events, orders, max_seq, targets, labels) in tqdm(enumerate(self.loaders["train"])):
                # to device
                for value in events.values():
                    for key in value:
                        value[key] = value[key].to(self.args.device)
                labels = labels.to(device=self.args.device, dtype=torch.float32)
                # forward
                pred, loss = self.model(events, orders, max_seq, self.cfg["datarc"]["batch_size"], targets, labels)
                # backward
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg["runner"]["gradient_clipping"])
                if not math.isnan(grad_norm):
                    self.optim.step()
                else:
                    print(f"\nnan in batch {batch_id}")
                pbar.set_postfix({"loss": f"{loss.item():4.2f}"})
                total_loss += loss.item() * len(labels)
            total_loss /= len(self.datasets["train"])
            self.valid()
        probs, predict_alert_keys = self.inference()
        
    def valid(self):
        self.best_valid_precision = getattr(self, "best_valid_precision", 0)
        self.model.eval()
        total_loss = 0
        all_outs = []
        all_labels = []
        with torch.no_grad():
            for events, orders, max_seq, targets, labels in tqdm(self.loaders["valid"]):
                # to device
                for value in events.values():
                    for key in value:
                        value[key] = value[key].to(self.args.device)
                all_labels.extend(labels.tolist())
                labels = labels.to(device=self.args.device, dtype=torch.float32)
                # forward
                out, loss = self.model(events, orders, max_seq, self.cfg["datarc"]["batch_size"], targets, labels)
                all_outs.extend(out.cpu().tolist())
                total_loss += loss.item() * len(labels)
        # sort by outs
        all_outs = enumerate(sorted(zip(all_outs, all_labels), key=lambda x: x[0], reverse=True))
        # get the reported part
        all_outs = list(filter(lambda x: x[1][1]==1, all_outs))
        total_loss /= len(self.loaders["valid"].dataset)
        
        valid_precision = (len(all_outs) - 1) / all_outs[-2][0]
        if self.best_valid_precision < valid_precision:
            self.best_valid_precision = valid_precision
            self._save(self.args.expdir / "val_best.ckpt")
        
        print(f"\nvalid loss:{total_loss:8.6f}\nprecision :{valid_precision:8.6f}")
        
    def inference(self):
        self._load(self.args.expdir / "val_best.ckpt")
        self.model.eval()
        probs = []
        predict_alert_keys = []
        with torch.no_grad():
            for events, orders, max_seq, targets, alert_keys in tqdm(self.loaders['test']):
                predict_alert_keys.extend(alert_keys)
                for value in events.values():
                    for key in value:
                        value[key] = value[key].to(self.args.device)
                out = self.model(events, orders, max_seq, self.cfg["datarc"]["batch_size"], targets)
                out = torch.sigmoid(out)
                probs.extend(out.cpu().tolist())
            
        self.write_prediction(probs, predict_alert_keys, self.args.prediction_filename)
        return probs, predict_alert_keys
        
    def write_prediction(self, probs, predict_alert_keys, filename="prediction.csv"):
        # get all alert keys which should be predicted
        all_alert_keys = set(pd.read_csv(self.args.data_path / "submission_sample.csv").alert_key)
        other_alert_keys = all_alert_keys - set(predict_alert_keys)
        
        # write file
        lines = sorted(zip(predict_alert_keys, probs), key=lambda x: x[1], reverse=True)
        lines = "\n".join(chain(
            map(lambda x: f"{x[0]},{x[1]}", lines),
            map(lambda x: f"{x},0.0", other_alert_keys),
        ))
        with open(self.args.expdir / filename, 'w') as f:
            f.write("alert_key,probability\n")
            f.write(lines)
            f.write("\n")
            
    def _save(self, path):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "args": self.args,
            "config": self.cfg,
        }
        torch.save(ckpt, path)
        
    def _load(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])
        if hasattr(self, "optim"):
            self.optim.load_state_dict(ckpt["optimizer"])
        self.args = ckpt["args"]
        self.cfg = ckpt["config"]
        