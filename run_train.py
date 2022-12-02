import os
import yaml
import pickle
import argparse
from pathlib import Path
from multiprocessing import cpu_count

import torch

from src.runner import Runner

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--runner_config', help='Config of runner', default='config/runner_config.yaml')
    parser.add_argument('-d', '--device', choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument('-n', "--expname", help="experiment name", default="tmp")
    parser.add_argument('--data_path', default="data")
    parser.add_argument('-a', '--amp', action="store_true", help="use fp16")
    parser.add_argument('-m', "--mode", choices=['train', 'valid', 'inference'], default="train")
    parser.add_argument('-t', "--split", default="all")
    parser.add_argument('-s', "--seed", default=1337, type=int)
    parser.add_argument('--deterministic', type=bool, default=False)
    parser.add_argument('--prediction_filename', default="prediction.csv")
    
    args = parser.parse_args()
    args.expdir = Path(f"runs/{args.expname}")
    if args.expdir.exists():
        pass
        try:
            with open(args.expdir / "config/args.pkl", 'rb') as f:
                args = pickle.load(f)
            with open(args.expdir / "config/runner_config.yaml", 'rb') as f:
                config = yaml.load(f, yaml.Loader)
        except:
            pass
    else:
        pass
        os.makedirs(args.expdir / "config", exist_ok=True)
        # with open(args.expdir / "config/args.pkl", 'wb') as f:
        #     pickle.dump(args, f)
        # with open(args.expdir / "config/finetune_runner.yaml", 'wb') as f:
        #         yaml.dump(config, f)
    
    with open(args.runner_config, 'rb') as f:
        config = yaml.load(f, yaml.Loader)
        
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if config['datarc']["num_workers"] == "auto":
        config['datarc']["num_workers"] = cpu_count()
    if args.split == "all":
        args.split = ["train", "valid", "test"]
    else:
        args.split = args.split.split(",")
    args.data_path = Path(args.data_path)
        
    return args, config

if __name__ == "__main__":
    args, runner_config = getargs()
    runner = Runner(args, runner_config)
    getattr(runner, args.mode)()