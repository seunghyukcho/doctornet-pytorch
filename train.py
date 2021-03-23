import json

import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.backends import cudnn
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from model import DoctorNet
from dataset import LabelMeDataset
from argument import add_model_args, add_train_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_model_args(parser)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    print('Loading train dataset...')
    train_dataset = LabelMeDataset(args.train_data, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    print('Loading validation dataset...')
    valid_dataset = LabelMeDataset(args.valid_data)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    print('Building model...')
    model = DoctorNet(args.n_classes, args.n_annotators, args.weight_type, args.feature_dim, args.bottleneck_dim)
    model = model.to(args.device)

    # Ignore annotators labeling which is -1
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    # Freeze feature extractor
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    optimizer_doctornet = Adam([model.annotators], lr=args.dn_lr, weight_decay=args.dn_l2)

    print('Start DoctorNet training!')
    best_model = copy.deepcopy(model)
    best_accuracy = 0
    writer = SummaryWriter(args.log_dir)
    for epoch in range(1, args.dn_epochs + 1):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for x, y, annotation in train_loader:
            model.zero_grad()

            x, y, annotation = x.to(args.device), y.to(args.device), annotation.to(args.device)
            pred, _ = model(x)

            pred = pred.view(-1, args.n_classes)
            annotation = annotation.view(-1)
            loss = criterion(pred, annotation)

            loss.backward()
            optimizer_doctornet.step()
            train_loss += loss.item()

            pred = pred.view(-1, args.n_annotators, args.n_classes)
            pred = torch.sum(pred, axis=1)
            pred = torch.argmax(pred, dim=1)
            train_correct += torch.sum(torch.eq(pred, y)).item()

        # Validation
        with torch.no_grad():
            valid_correct = 0
            model.eval()
            for x, y in valid_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x, pred=True)
                pred = torch.argmax(pred, dim=1)
                valid_correct += torch.sum(torch.eq(pred, y)).item()

        print(
            f'Epoch: {(epoch):4d} | '
            f'Train Loss: {train_loss:.3f} | '
            f'Train Accuracy: {(train_correct / len(train_dataset)):.2f} | '
            f'Valid Accuracy: {(valid_correct / len(valid_dataset)):.2f}'
        )

        # Save tensorboard log
        if epoch % args.log_interval == 0:
            writer.add_scalar('dn_train_loss', train_loss, epoch)
            writer.add_scalar('dn_train_accuracy', train_correct / len(train_dataset), epoch)
            writer.add_scalar('dn_valid_accuracy', valid_correct / len(valid_dataset), epoch)

        if best_accuracy < valid_correct:
            best_accuracy = valid_correct
            best_model = copy.deepcopy(model)

    # Freeze DoctorNet
    model.annotators.requires_grad = False
    optimizer_weight = Adam(model.weights.parameters(), lr=args.aw_lr, weight_decay=args.aw_l2)

    print('\n\nStart DoctorNet averaging weight training!')
    model = best_model
    best_accuracy = 0
    for epoch in range(1, args.aw_epochs + 1):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for x, y, annotation in train_loader:
            model.zero_grad()

            x, y, annotation = x.to(args.device), y.to(args.device), annotation.to(args.device)
            decisions, weights = model(x, weight=True)

            # Calculate loss of annotators' labeling
            mask = annotation != -1

            # Calculate sum of one-hot encoded anotators' labels
            annotation = annotation + 1
            annotation = F.one_hot(annotation)
            annotation = annotation[:, :, 1:].float()
            annotation = torch.sum(annotation, axis=1) / torch.sum((annotation != -1), axis=1)
            annotation = torch.argmax(annotation, dim=1)
            
            pred = torch.sum(decisions, axis=1)
            decisions = decisions.masked_fill(mask[:, :, None], 0)
            decisions = torch.sum(decisions, axis=1)

            weights = weights.masked_fill(mask, 0)
            weights = torch.sum(weights, axis=-1)

            decisions = decisions / weights[:, None]
            loss = criterion(decisions, annotation)

            loss.backward()
            optimizer_weight.step()
            train_loss += loss.item()

            pred = torch.argmax(pred, dim=1)
            train_correct += torch.sum(torch.eq(pred, y)).item()

        # Validation
        with torch.no_grad():
            valid_correct = 0
            model.eval()
            for x, y in valid_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x, pred=True, weight=True)
                pred = torch.argmax(pred, dim=1)
                valid_correct += torch.sum(torch.eq(pred, y)).item()

        print(
            f'Epoch: {(epoch):4d} | '
            f'Train Loss: {train_loss:.3f} | '
            f'Train Accuracy: {(train_correct / len(train_dataset)):.2f} | '
            f'Valid Accuracy: {(valid_correct / len(valid_dataset)):.2f}'
        )

        # Save tensorboard log
        if epoch % args.log_interval == 0:
            writer.add_scalar('aw_train_loss', train_loss, epoch)
            writer.add_scalar('aw_train_accuracy', train_correct / len(train_dataset), epoch)
            writer.add_scalar('aw_valid_accuracy', valid_correct / len(valid_dataset), epoch)

        # Save the model with highest accuracy on validation set
        if best_accuracy < valid_correct:
            best_accuracy = valid_correct
            checkpoint_dir = Path(args.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': model.state_dict()
            }, checkpoint_dir / 'best_model.pth')

            with open(checkpoint_dir / 'args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

