import pickle
import random
import datetime
import argparse

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from config import Config
from logger import Logger
from models import RegressionOptimizer


def get_model_fname(cfg):
    rand_str = str(int(random.random() * 9e6))
    model_fname = "-".join([
        datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S"),
        rand_str,
        cfg['cfg_id']
    ])
    return model_fname


def setup_data_loader(cfg, data):
    X = data[0]
    y = data[1]

    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)

    dataset = TensorDataset(tensor_x, tensor_y)

    batch_size = cfg['model']['batch_size']
    train_eval_split = cfg['model']['train_eval_split']
    train_size = int(train_eval_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=cfg['model']['shuffle_data'],
        num_workers=0
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0
    )

    return train_data_loader, test_data_loader


def run_experiment(cfg_id, n_runs=1):
    cfg = Config().get_cfg(cfg_id)
    with open("datasets/" + cfg['dataset_id'] + "data.npy", "rb") as f:
        data = pickle.load(f)

    train_data_loader, test_data_loader = setup_data_loader(cfg, data)

    for _ in range(n_runs):
        model_fname = get_model_fname(cfg)
        logger = Logger(model_fname, cfg)
        logger.log_config()
        optimizer = RegressionOptimizer(
            cfg, train_data_loader, test_data_loader, logger
        )
        optimizer.train()
        logger.close()


def gen_data(cfg_id):
    cfg = Config().get_cfg(cfg_id)
    high2 = cfg['problems']['high2']
    n = cfg['problems']['n_problems']

    X = np.random.randint(1, high2, size=(n, 2))
    X = np.unique(X, axis=0)
    y = X[:, 0] * X[:, 1]

    # TODO: Do this better..
    X = X.astype(float)
    y = y.astype(float)

    if cfg['problems']['log_norm']:
        X = np.log(X)
        y = np.log(y)

        max_val = max(X.max(), y.max())
        print(max_val)
        X /= max_val
        y /= max_val
    else:
        X /= high2
        y /= high2 * 10.

    print(X)
    print(y)

    with open("datasets/" + cfg['cfg_id'] + "data.npy", "wb") as f:
        pickle.dump((X, y), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gendata", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("-c", "--cfg_id", nargs=None, help="cfg_id")
    parser.add_argument("-n", "--nruns", nargs="?", type=int, default=1)
    args = parser.parse_args()

    if args.gendata:
        gen_data(args.cfg_id)
    elif args.train:
        run_experiment(args.cfg_id, args.nruns)
