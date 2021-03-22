import json
import torch
import argparse
import importlib
from tqdm import tqdm
from pathlib import Path
from munch import munchify
from torch.utils.data import DataLoader

from argument import add_test_args
from dataset import LabelMeDataset
from model import DoctorNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    args = parser.parse_args()

    print('Loading configurations...')
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_file = ckpt_dir / 'best_model.pth'
    ckpt = torch.load(ckpt_file)

    config_file = ckpt_dir / 'args.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = munchify(config)

    test_dataset = LabelMeDataset(args.test_data)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    model = DoctorNet(config.n_classes, config.n_annotators, config.weight_type, config.feature_dim, config.bottleneck_dim)
    model.load_state_dict(ckpt['model'])
    model = model.to(args.device)

    with torch.no_grad():
        model.eval()
        correct = 0
        for x, y in tqdm(test_loader):
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x, pred=True, weight=True)
            pred = torch.argmax(pred, dim=1)
            correct += torch.sum(torch.eq(pred, y)).item()

        print(f'Test Accuracy: {correct / len(test_dataset)}')
