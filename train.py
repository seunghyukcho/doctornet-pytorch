import json

import copy
import torch
import argparse
import importlib
import numpy as np
import torch.nn as nn
from torch.backends import cudnn
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
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
    train_dataset = LabelMeDataset(args.train_dir, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    print('Loading validation dataset...')
    valid_dataset = LabelMeDataset(args.valid_dir)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    print('Building model...')
    model = DoctorNet(args.n_classes, args.n_annotators, args.weight_type, args.feature_dim, args.bottleneck_dim)
    model = model.to(args.device)

    # Ignore annotators labeling which is -1
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Start DoctorNet training!')
    best_model = copy.deepcopy(model)
    best_loss = 1e9
    writer = SummaryWriter(args.log_dir)
    for epoch in range(1, args.doctornet_epochs + 1):
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
            optimizer.step()
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

        if best_loss > train_loss:
            best_model = model
            best_model = copy.deepcopy(model)

    model = best_model
    print('\n\nStart DoctorNet Averaging Weight training!')
    best_accuracy = 0
    for epoch in range(1, args.weight_epochs + 1):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for x, y, annotation in train_loader:
            model.zero_grad()

            x, y, annotation = x.to(args.device), y.to(args.device), annotation.to(args.device)
            ann_out, cls_out = model(x, annotator)

            # Calculate loss of annotators' labeling
            ann_out = torch.reshape(ann_out, (-1, args.n_class))
            annotation = annotation.view(-1)
            loss = criterion(ann_out, annotation)

            # Regularization term
            confusion_matrices = model.noise_adaptation_layer
            matrices = confusion_matrices.local_confusion_matrices - confusion_matrices.global_confusion_matrix
            for matrix in matrices:
                loss -= args.scale * torch.linalg.norm(matrix)

            # Update model weight using gradient descent
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate classifier accuracy
            pred = torch.argmax(cls_out, dim=1)
            train_correct += torch.sum(torch.eq(pred, y)).item()

        # Validation
        with torch.no_grad():
            valid_correct = 0
            model.eval()
            for x, y in valid_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x)
                pred = torch.argmax(pred, dim=1)
                valid_correct += torch.sum(torch.eq(pred, y)).item()

        print(
            f'Epoch: {(epoch + 1):4d} | '
            f'Train Loss: {train_loss:.3f} | '
            f'Train Accuracy: {(train_correct / len(train_dataset)):.2f} | '
            f'Valid Accuracy: {(valid_correct / len(valid_dataset)):.2f}'
        )

        # Save tensorboard log
        if epoch % args.log_interval == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_accuracy', train_correct / len(train_dataset), epoch)
            writer.add_scalar('valid_accuracy', valid_correct / len(valid_dataset), epoch)

        # Save the model with highest accuracy on validation set
        if best_accuracy < valid_correct:
            best_accuracy = valid_correct
            checkpoint_dir = Path(args.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'auxiliary_network': model.auxiliary_network.state_dict(),
                'noise_adaptation_layer': model.noise_adaptation_layer.state_dict(),
                'classifier': model.classifier.state_dict()
            }, checkpoint_dir / 'best_model.pth')

            with open(checkpoint_dir / 'args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

