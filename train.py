from loader import *
import os
from sklearn.metrics import f1_score
import pandas as pd
pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import timm
from sklearn.model_selection import StratifiedKFold

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_model(model, optimizer, scheduler, fold, epoch, best=False):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    if not (os.path.isdir('./saved_model')): os.mkdir('./saved_model')
    if not (os.path.isdir('./saved_model/{}'.format(model_arch))): os.mkdir('./saved_model/{}'.format(model_arch))
    torch.save(state, './saved_model/{}/fold_{}_epoch_{}'.format(model_arch, fold+1, epoch+1))
    if best == True:
        if not (os.path.isdir('./best_model')): os.mkdir('./best_model')
        if not (os.path.isdir('./best_model/{}'.format(model_arch))): os.mkdir('./best_model/{}'.format(model_arch))
        torch.save(state, './best_model/{}/fold_{}_epoch_{}'.format(model_arch, fold+1, epoch+1))

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, optimizer, scheduler, fold, epoch):
        if self.val_loss_min == np.Inf:
            self.val_loss_min = val_loss
        elif val_loss > self.val_loss_min:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                print('Early Stopping - Fold {} Training is Stopping'.format(fold))
                self.early_stop = True
        else:  # val_loss < val_loss_min
            save_model(model, optimizer, scheduler, fold, epoch, best=True)
            print('*** Validation loss decreased ({} --> {}).  Saving model... ***'.\
                  format(round(self.val_loss_min, 6), round(val_loss, 6)))
            self.val_loss_min = val_loss
            self.counter = 0

class Model(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if model_name=='tf_efficientnet_b4_ns':
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
        else: # 'resnet50', 'resnext50_32x4d'
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler):
    model.train()
    lst_out = []
    lst_label = []
    avg_loss = 0
    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device).float()
        labels = labels.to(device).long()
        with autocast():
            preds = model(images)
            lst_out += [torch.argmax(preds, 1).detach().cpu().numpy()]
            lst_label += [labels.detach().cpu().numpy()]

            loss = loss_fn(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            avg_loss += loss.item() / len(train_loader)
    scheduler.step()
    lst_out = np.concatenate(lst_out); lst_label = np.concatenate(lst_label)
    accuracy = (lst_out == lst_label).mean()
    print('{} epoch - train loss : {}, train accuracy : {}'.\
          format(epoch + 1, np.round(avg_loss, 6), np.round(accuracy*100, 2)))

def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    model.eval()
    lst_val_out = []
    lst_val_label = []
    avg_val_loss = 0
    for step, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        val_images = images.to(device).float()
        val_labels = labels.to(device).long()

        val_preds = model(val_images)
        lst_val_out += [torch.argmax(val_preds, 1).detach().cpu().numpy()]
        lst_val_label += [val_labels.detach().cpu().numpy()]
        val_loss = loss_fn(val_preds, val_labels)
        avg_val_loss += val_loss.item() / len(val_loader)
    lst_val_out = np.concatenate(lst_val_out); lst_val_label = np.concatenate(lst_val_label)
    accuracy = (lst_val_out == lst_val_label).mean()
    print('{} epoch - valid loss : {}, valid f1 score : {}'.\
          format(epoch + 1, np.round(avg_val_loss, 6), np.round(accuracy*100,2)))
    return avg_val_loss


if __name__ == "__main__":
    batch_size = 32
    num_workers = 4
    seed = 42
    split = 5
    epochs = 100
    patience = 5
    # sample_num = (200000, 100000) # train, valid
    model_arch = 'resnet50' # 'resnet50', 'resnext50_32x4d', 'tf_efficientnet_b4_ns'
    model_weight = './packages/resnet50-19c8e357.pth' # 'resnet50-19c8e357.pth', 'resnext50_32x4d-7cdf4587.pth'

    trainval_dataset = pd.read_csv('../DATA/trainset-for_user.csv', header=None)
    X_train = trainval_dataset.iloc[:, :-1]
    Y_train = trainval_dataset.iloc[:, -1]
    seed_everything(seed)
    cv = StratifiedKFold(n_splits=split, random_state=seed, shuffle=True)
    for fold, (train_index, val_index) in enumerate(cv.split(X_train, Y_train)):
        print('---------- Fold {} is training ----------'.format(fold + 1))
        # train_index = np.random.choice(train_index, sample_num[0], replace=False)
        # val_index = np.random.choice(val_index, sample_num[1], replace=False)
        train_x, train_y = X_train.iloc[train_index], Y_train[train_index]
        val_x, val_y = X_train.iloc[val_index], Y_train[val_index]

        train_dataset = TrainDataset(train_x, train_y)
        val_dataset = TrainDataset(val_x, val_y)
        train_loader = data_loader('train', train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_loader = data_loader('valid', val_dataset, batch_size=batch_size, num_workers=num_workers)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(model_name=model_arch).to(device)
        model.load_state_dict(torch.load(model_weight), strict=False)
        loss_tr = nn.CrossEntropyLoss().to(device); loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        scaler = GradScaler()
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler)
            save_model(model, optimizer, scheduler, fold, epoch)
            with torch.no_grad():
                val_loss = valid_one_epoch(epoch, model, loss_fn, val_loader, device)
                early_stopping(val_loss, model, optimizer, scheduler, fold, epoch)
                if early_stopping.early_stop:
                    break

        del model, optimizer, train_dataset, val_dataset, train_loader, val_loader, scheduler, scaler
        torch.cuda.empty_cache()